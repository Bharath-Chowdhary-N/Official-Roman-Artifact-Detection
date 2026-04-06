"""
upload_pipeline.py

Combined script that:
  1. Uploads a panel-5 FITS file to Zooniverse as a two-frame subject
  2. Immediately generates a Caesar annotation CSV for that subject
  3. Uploads the CSV to Caesar as pre-loaded editable bounding boxes

Coordinate alignment fix
-------------------------
FITS/pipeline coordinates use origin='lower' (y=0 at bottom).
Zooniverse displays images with origin='upper' (y=0 at top).

Instead of mathematically flipping y coordinates, we render BOTH images
(clean and annotated) with origin='upper' so the image coordinate system
matches Zooniverse naturally. The FITS detection coordinates are then used
directly in Caesar without any transformation.

Subject frames:
  Frame 0 — clean image (rendered with origin='upper', volunteers draw on this)
  Frame 1 — annotated image (rendered with origin='upper', ML boxes for reference)

Caesar annotation format (confirmed by Zooniverse team):
  x_center  = centroid_x
  y_center  = image_height - centroid_y   (flip: FITS y=0 at bottom → Zooniverse y=0 at top)
  width     = x_max - x_min
  height    = y_max - y_min
  toolIndex = 0 (confident), 1 (suspicious), 2 (confirmed ZO)
  taskIndex = 0
  stepKey   = "S0"

Usage
-----
# Single file:
python upload_pipeline.py \\
    --fits   Images/output/EUC_..._panel5.fits \\
    --detections  Images/output/EUC_..._detections.csv \\
    --subject-set-id 134713 \\
    --workflow-id    31349  \\
    --username YOUR_USER --password YOUR_PASS

# Directory mode:
python upload_pipeline.py \\
    --fits-dir Images/output/ \\
    --detections-dir Images/output/ \\
    --subject-set-id 134713 \\
    --workflow-id    31349  \\
    --username YOUR_USER --password YOUR_PASS

Credentials can also be set via env vars:
    ZOONIVERSE_USERNAME
    ZOONIVERSE_PASSWORD
"""

import argparse
import json
import os
import random
import string
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import requests
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from panoptes_client import Panoptes, Subject, SubjectSet  # type: ignore


# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------

# Must match Zooniverse Project Builder T0 tool order:
#   Tool 0 = Confident detection  (Green)
#   Tool 1 = Suspicious detection (Yellow)
#   Tool 2 = Confirmed zero-order (Green)
DETECTION_TYPE_TO_TOOL: dict[str, int] = {
    "confident":    0,
    "suspicious":   1,
    "confirmed ZO": 2,
}

EXTRACTOR_KEY = "machineLearnt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_mark_id(n: int = 6) -> str:
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(n))


def det_code_to_idx(code: str) -> int:
    return (int(code[1]) - 1) * 4 + (int(code[0]) - 1)


def zscale(arr: np.ndarray) -> np.ndarray:
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(arr[np.isfinite(arr)])
    out = np.clip((arr - vmin) / (vmax - vmin + 1e-30), 0, 1)
    out[~np.isfinite(arr)] = 0
    return out


# ---------------------------------------------------------------------------
# Step 1: Render and upload subject to Zooniverse
# ---------------------------------------------------------------------------

def render_clean_png(image_data: np.ndarray, png_path: str, dpi: int = 300) -> None:
    """
    Render clean image with origin='lower' (FITS convention, y=0 at bottom).
    Zooniverse will display it as-is, and FITS coordinates map directly
    onto the image without any transformation.
    """
    disp = zscale(image_data)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(disp, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0, format='png')
    plt.close(fig)


def render_annotated_png(
    image_data: np.ndarray,
    det_table,
    zx, zy,
    image_height: int,
    png_path: str,
    dpi: int = 300,
) -> None:
    """
    Render annotated image with origin='lower' (FITS convention).
    Bounding box coordinates are used directly without any transformation.
    """
    disp = zscale(image_data)

    confident  = [r for r in det_table if r['TYPE'].strip() == 'confident']
    suspicious = [r for r in det_table if r['TYPE'].strip() == 'suspicious']
    confirmed  = [r for r in det_table if r['TYPE'].strip() == 'confirmed ZO']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(disp, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

    def _add_boxes(objects, colour):
        for r in objects:
            x0, y0 = int(r['X_MIN']), int(r['Y_MIN'])
            x1, y1 = int(r['X_MAX']), int(r['Y_MAX'])
            ax.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.5, edgecolor=colour, facecolor='none', alpha=0.9,
            ))

    _add_boxes(confident,  'lime')
    _add_boxes(suspicious, 'yellow')
    _add_boxes(confirmed,  'blue')

    if len(zx):
        ax.scatter(zx, zy, marker='s', facecolors='none',
                   edgecolors='red', linewidths=1.5, s=40)

    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0, format='png')
    plt.close(fig)


def find_existing_subject(subject_set: SubjectSet, filename: str) -> int | None:
    """
    Return the subject_id if a subject with this filename already exists
    in the set, otherwise None.  Scans up to 500 subjects to avoid long waits.
    """
    count = 0
    for subject in subject_set.subjects:
        if subject.metadata.get("filename") == filename:
            return int(subject.id)
        count += 1
        if count >= 500:
            break
    return None


def upload_subject(
    fits_path: str,
    subject_set: SubjectSet,
    dpi: int = 300,
) -> tuple[int, str, int]:
    """
    Upload FITS to Zooniverse as a two-frame subject.
    Returns (subject_id, fits_basename, image_height).
    If a subject with the same filename already exists in the set, it is
    reused instead of creating a duplicate.
    """
    print(f"\n--- Uploading subject ---")
    print(f"  FITS: {fits_path}")

    with fits.open(fits_path) as hdul:
        preproc_names = [h.name for h in hdul if h.name.endswith('.PREPROCESSED')]
        if not preproc_names:
            raise ValueError(f"No .PREPROCESSED extension in {fits_path}")
        preproc_name = preproc_names[0]
        det_code  = preproc_name.split('.')[0][3:]
        det_idx   = det_code_to_idx(det_code)
        det_label = f"DET{det_code}"

        image_data = hdul[preproc_name].data.astype(np.float64)
        sci_header = hdul[preproc_name].header
        obs_id     = sci_header.get('OBS_ID',  'unknown')
        grism      = sci_header.get('GWA_POS', 'unknown')
        image_height = image_data.shape[0]

        det_name = f"{det_label}.DETECTIONS"
        if det_name not in hdul:
            raise ValueError(f"No {det_name} table in {fits_path}")
        det_table = hdul[det_name].data

        zo_data = hdul['ZOTABLE'].data
        zx = zo_data['X'][zo_data['Detector'] == det_idx]
        zy = zo_data['Y'][zo_data['Detector'] == det_idx]

    confident  = [r for r in det_table if r['TYPE'].strip() == 'confident']
    suspicious = [r for r in det_table if r['TYPE'].strip() == 'suspicious']
    confirmed  = [r for r in det_table if r['TYPE'].strip() == 'confirmed ZO']

    print(f"  Image size  : {image_height} x {image_data.shape[1]}")
    print(f"  Detections  : {len(confident)} confident | "
          f"{len(suspicious)} suspicious | {len(confirmed)} confirmed ZO")

    # Build metadata
    def _bbox_list(objects, label):
        return [
            {
                'type':   label,
                'x_min':  int(r['X_MIN']),
                'y_min':  int(r['Y_MIN']),
                'x_max':  int(r['X_MAX']),
                'y_max':  int(r['Y_MAX']),
                'cent_x': float(r['CENT_X']),
                'cent_y': float(r['CENT_Y']),
                'area':   int(r['AREA']),
                'fill':   float(r['FILL']),
                'aspect': float(r['ASPECT']),
            }
            for r in objects
        ]

    all_boxes = (
        _bbox_list(confident,  'confident') +
        _bbox_list(suspicious, 'suspicious') +
        _bbox_list(confirmed,  'confirmed_ZO')
    )

    metadata = {
        'filename':        os.path.basename(fits_path),
        'detector':        det_label,
        'obs_id':          str(obs_id),
        'grism':           str(grism),
        'n_confident':     len(confident),
        'n_suspicious':    len(suspicious),
        'n_confirmed_zo':  len(confirmed),
        'n_zo_positions':  len(zx),
        '#bounding_boxes': json.dumps(all_boxes),
        '#zo_positions':   json.dumps([{'x': float(x), 'y': float(y)}
                                       for x, y in zip(zx, zy)]),
    }

    # Look for existing clean image generated by main.py/full_pipeline_5panel.py
    # Saved as <fits_stem>_panel5_clean.jpg or .png
    existing_clean = None
    for ext in ('.jpg', '.jpeg', '.png'):
        candidate = fits_path.replace('_panel5.fits', f'_panel5_clean{ext}')
        if os.path.exists(candidate):
            existing_clean = candidate
            break

    if existing_clean:
        print(f"  Using existing clean image: {existing_clean}")
        clean_path = existing_clean
        tmp_clean_path = None
    else:
        print(f"  No existing clean image found, rendering fresh (origin=lower) ...")
        tmp = tempfile.NamedTemporaryFile(suffix='_clean.png', delete=False)
        clean_path = tmp.name
        tmp.close()
        tmp_clean_path = clean_path
        render_clean_png(image_data, clean_path, dpi=dpi)

    # Always render annotated image fresh (origin=lower)
    tmp_anno = tempfile.NamedTemporaryFile(suffix='_anno.png', delete=False)
    anno_path = tmp_anno.name
    tmp_anno.close()

    try:
        # Reuse existing subject if already uploaded (avoids duplicates)
        existing_id = find_existing_subject(subject_set, metadata['filename'])
        if existing_id is not None:
            print(f"  Subject already exists (ID {existing_id}) — skipping upload")
            return existing_id, metadata['filename'], image_height

        print(f"  Rendering annotated image (origin=lower) ...")
        render_annotated_png(image_data, det_table, zx, zy,
                             image_height, anno_path, dpi=dpi)

        subject = Subject()
        subject.links.project = subject_set.links.project
        subject.add_location(clean_path)   # Frame 0: clean (from main.py)
        subject.add_location(anno_path)    # Frame 1: annotated
        subject.metadata.update(metadata)
        subject.save()
        subject_set.add([subject])

        subject_id = int(subject.id)
        print(f"  Uploaded subject ID : {subject_id} (2 frames)")
        return subject_id, metadata['filename'], image_height

    finally:
        # Only delete clean image if we rendered it fresh
        if tmp_clean_path and os.path.exists(tmp_clean_path):
            os.remove(tmp_clean_path)
        if os.path.exists(anno_path):
            os.remove(anno_path)


# ---------------------------------------------------------------------------
# Step 2: Build Caesar CSV (no y-flip needed — images already flipped)
# ---------------------------------------------------------------------------

def build_caesar_csv(
    detections_csv: str,
    subject_id: int,
    image_height: int,
    output_csv: str,
) -> int:
    """
    Build Caesar annotation CSV for a specific subject.
    Y-axis is flipped: FITS uses origin=lower (y=0 at bottom),
    but Zooniverse displays images with origin=upper (y=0 at top).
    Corrected y_center = image_height - centroid_y
    """
    print(f"\n--- Building Caesar CSV ---")
    detections = pd.read_csv(detections_csv)

    mark_data = []
    for _, row in detections.iterrows():
        tool_index = DETECTION_TYPE_TO_TOOL.get(row["type"])
        if tool_index is None:
            continue

        x1, y1 = float(row["x_min"]), float(row["y_min"])
        x2, y2 = float(row["x_max"]), float(row["y_max"])

        mark_data.append({
            "stepKey":   "S0",
            "taskIndex": 0,
            "toolIndex": tool_index,
            "x_center":  float(row["centroid_x"]),
            "y_center":  image_height - float(row["centroid_y"]),  # flip: FITS y=0 at bottom → Zooniverse y=0 at top
            "width":     x2 - x1,
            "height":    y2 - y1,
        })

    new_row = pd.DataFrame([{
        "subject_id":    subject_id,
        "extractor_key": EXTRACTOR_KEY,
        "data":          json.dumps({"data": mark_data}),
    }])

    # Accumulate: merge with existing CSV so Caesar always receives the full
    # history.  If this subject was already in the file, replace its row.
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        existing = existing[existing["subject_id"] != subject_id]
        df = pd.concat([existing, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(output_csv, index=False)

    print(f"  Subject ID  : {subject_id}")
    print(f"  Total marks : {len(mark_data)}")
    print(f"  Total subjects in CSV : {len(df)}")
    print(f"  CSV written : {output_csv}")
    return len(mark_data)


# ---------------------------------------------------------------------------
# Step 3: Upload to Caesar via 0x0.st temporary file host
# ---------------------------------------------------------------------------

def upload_to_caesar(workflow_id: str, csv_path: str) -> bool:
    """
    Upload the CSV to 0x0.st (a plain HTTP file host with no splash page),
    then POST the returned public URL to Caesar.
    Caesar only accepts a URL — direct multipart upload is not supported.
    Returns True on success.
    """
    print(f"\n--- Uploading to Caesar ---")

    with open(csv_path, 'rb') as f:
        csv_content = f.read()

    # Step 3a: push CSV to a temporary public file host
    csv_url = None
    fname = os.path.basename(csv_path)

    # Try file.io
    print("  Uploading CSV to file.io ...")
    try:
        r = requests.post(
            "https://file.io/?expires=1h",
            files={"file": (fname, csv_content, "text/csv")},
            timeout=30,
        )
        if r.status_code in (200, 201):
            data = r.json()
            if data.get("success"):
                csv_url = data["link"]
    except Exception as e:
        print(f"  file.io failed: {e}")

    # Fallback: transfer.sh
    if not csv_url:
        print("  Falling back to transfer.sh ...")
        try:
            r = requests.put(
                f"https://transfer.sh/{fname}",
                data=csv_content,
                headers={"Content-Type": "text/csv"},
                timeout=30,
            )
            if r.status_code == 200:
                csv_url = r.text.strip()
        except Exception as e:
            print(f"  transfer.sh failed: {e}")

    if not csv_url:
        print("  ERROR: all file hosting services failed")
        return False

    print(f"  Public URL: {csv_url}")
    print(f"  Public URL: {csv_url}")

    # Step 3b: hand the URL to Caesar
    token = Panoptes.client().get_bearer_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    response = requests.post(
        f"https://caesar.zooniverse.org/workflows/{workflow_id}/extracts/import",
        json={"file": csv_url},
        headers=headers,
    )

    print(f"  Status: {response.status_code}")
    if response.status_code == 204:
        print("  SUCCESS - extracts accepted by Caesar (HTTP 204)")
        return True
    else:
        print(f"  ERROR - {response.text[:500]}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="upload_pipeline",
        description=(
            "Upload FITS to Zooniverse and push ML detections to Caesar "
            "as editable bounding boxes in one step."
        ),
    )

    # Input — single file mode
    parser.add_argument("--fits", "-i",
                        help="Single panel5 FITS file to upload")
    parser.add_argument("--detections", "-d",
                        help="Single *_detections.csv matching --fits")

    # Input — directory mode
    parser.add_argument("--fits-dir",
                        help="Directory containing *_panel5.fits files")
    parser.add_argument("--detections-dir",
                        help="Directory containing *_detections.csv files")

    # Zooniverse / Caesar
    parser.add_argument("--subject-set-id", "-s", type=int, required=True)
    parser.add_argument("--workflow-id", "-w", required=True)
    parser.add_argument("--output", "-o", default="caesar_annotations.csv")
    parser.add_argument("--gdrive-file-id", "-g", default=None,
                        help="Google Drive file ID of the uploaded caesar_annotations.csv. "
                             "If provided, skips auto-hosting and sends this URL to Caesar. "
                             "Upload the CSV to Drive, set sharing to 'Anyone with the link', "
                             "and copy the ID from the share URL.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--username",
                        default=os.environ.get("ZOONIVERSE_USERNAME"))
    parser.add_argument("--password",
                        default=os.environ.get("ZOONIVERSE_PASSWORD"))
    args = parser.parse_args()

    if not args.username or not args.password:
        parser.error("Provide --username/--password or set "
                     "ZOONIVERSE_USERNAME/ZOONIVERSE_PASSWORD env vars")

    # Build list of (fits_path, detections_path) pairs
    pairs = []
    if args.fits and args.detections:
        pairs.append((args.fits, args.detections))
    elif args.fits_dir and args.detections_dir:
        for f in sorted(os.listdir(args.fits_dir)):
            if f.endswith('_panel5.fits'):
                fits_path = os.path.join(args.fits_dir, f)
                det_name  = f.replace('_panel5.fits', '_detections.csv')
                det_path  = os.path.join(args.detections_dir, det_name)
                if os.path.exists(det_path):
                    pairs.append((fits_path, det_path))
                else:
                    print(f"WARNING: no detections CSV for {f}, skipping")
    else:
        parser.error("Provide either --fits + --detections, "
                     "or --fits-dir + --detections-dir")

    print(f"Found {len(pairs)} FITS/detections pair(s) to process")

    # Connect
    print("\nConnecting to Panoptes ...")
    Panoptes.connect(
        username=args.username,
        password=args.password,
        endpoint="https://panoptes.zooniverse.org",
    )
    print("Connected.")
    subject_set = SubjectSet.find(args.subject_set_id)

    # Process each pair
    for fits_path, det_path in pairs:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(fits_path)}")
        print(f"{'='*60}")
        try:
            # Step 1: upload subject (images rendered with origin='upper')
            subject_id, fits_basename, image_height = upload_subject(
                fits_path, subject_set, dpi=args.dpi
            )

            # Step 2: build Caesar CSV (with y-axis flip)
            n_marks = build_caesar_csv(det_path, subject_id, image_height, args.output)

            # Step 3: upload to Caesar
            if n_marks > 0:
                if args.gdrive_file_id:
                    gdrive_url = f"https://drive.google.com/uc?export=download&id={args.gdrive_file_id}"
                    print(f"\n--- Uploading to Caesar (Google Drive) ---")
                    print(f"  URL: {gdrive_url}")
                    token = Panoptes.client().get_bearer_token()
                    resp = requests.post(
                        f"https://caesar.zooniverse.org/workflows/{args.workflow_id}/extracts/import",
                        json={"file": gdrive_url},
                        headers={"Authorization": f"Bearer {token}",
                                 "Content-Type": "application/json"},
                    )
                    print(f"  Status: {resp.status_code}")
                    if resp.status_code == 204:
                        print("  SUCCESS - extracts accepted by Caesar (HTTP 204)")
                    else:
                        print(f"  ERROR - {resp.text[:500]}")
                else:
                    upload_to_caesar(args.workflow_id, args.output)
            else:
                print("  No marks generated — skipping Caesar upload")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()