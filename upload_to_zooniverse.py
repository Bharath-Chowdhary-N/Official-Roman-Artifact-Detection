"""
Upload panel-5 FITS outputs to Zooniverse as two-frame subjects.

Each subject contains two images that volunteers can flip between:
  Frame 0 — clean image (no boxes, no axes) saved by main.py
  Frame 1 — annotated image (bounding boxes, no axes/title) rendered here

Bounding-box coordinates are stored as hidden metadata (#bounding_boxes)
for downstream data processing.

Usage:
    python upload_to_zooniverse.py <fits_file_or_dir> [options]

Credentials via env vars:
    ZOONIVERSE_USERNAME
    ZOONIVERSE_PASSWORD
or pass --username / --password explicitly.
"""

import os
import sys
import argparse
import json
import tempfile
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from panoptes_client import Panoptes, Subject, SubjectSet

SUBJECT_SET_ID = 134713


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def det_code_to_idx(code: str) -> int:
    """Convert a two-char detector code (e.g. '11') to a gelsa detector index."""
    return (int(code[1]) - 1) * 4 + (int(code[0]) - 1)


def zscale(arr: np.ndarray) -> np.ndarray:
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(arr[np.isfinite(arr)])
    out = np.clip((arr - vmin) / (vmax - vmin + 1e-30), 0, 1)
    out[~np.isfinite(arr)] = 0
    return out


def find_clean_image(fits_path: str) -> str | None:
    """
    Return the path to the clean panel-5 image saved by main.py, or None.
    Tries both .jpg and .png extensions.
    """
    base = fits_path.replace('_panel5.fits', '_panel5_clean')
    for ext in ('.jpg', '.png'):
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Render annotated PNG from FITS (no axes, no title)
# ---------------------------------------------------------------------------

def render_annotated_png(fits_path: str, png_path: str, dpi: int = 300) -> dict:
    """
    Render the annotated panel-5 image (boxes only, no axes or title).
    Returns subject metadata dict.
    """
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

    disp = zscale(image_data)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(disp, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

    def _add_boxes(objects, colour):
        for r in objects:
            x0, y0, x1, y1 = int(r['X_MIN']), int(r['Y_MIN']), int(r['X_MAX']), int(r['Y_MAX'])
            ax.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.5, edgecolor=colour, facecolor='none', alpha=0.9,
            ))

    _add_boxes(confident,  'lime')
    _add_boxes(suspicious, 'yellow')
    _add_boxes(confirmed,  'blue')
    if len(zx):
        ax.scatter(zx, zy, marker='s', facecolors='none', edgecolors='red',
                   linewidths=1.5, s=40)

    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0, format='png')
    plt.close(fig)

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

    return {
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


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_fits(fits_path: str, subject_set: SubjectSet,
                dpi: int = 300, dry_run: bool = False) -> None:
    print(f"\nProcessing: {fits_path}")

    clean_path = find_clean_image(fits_path)
    if clean_path is None:
        print("  WARNING: no clean image found — only the annotated frame will be uploaded")
    else:
        print(f"  Clean image:     {clean_path}")

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    annotated_path = tmp.name
    tmp.close()

    try:
        metadata = render_annotated_png(fits_path, annotated_path, dpi=dpi)
        print(f"  Annotated image: {annotated_path}")
        print(f"  Detections: {metadata['n_confident']} confident | "
              f"{metadata['n_suspicious']} suspicious | "
              f"{metadata['n_confirmed_zo']} confirmed ZO | "
              f"{metadata['n_zo_positions']} ZO positions")

        if dry_run:
            print("  [dry-run] skipping upload")
            return

        subject = Subject()
        subject.links.project = subject_set.links.project

        # Frame 0: clean (what volunteers classify on)
        # Frame 1: annotated (reference with machine detections)
        if clean_path:
            subject.add_location(clean_path)
        subject.add_location(annotated_path)

        subject.metadata.update(metadata)
        subject.save()
        subject_set.add([subject])
        print(f"  Uploaded subject ID: {subject.id}  "
              f"({'2 frames' if clean_path else '1 frame (annotated only)'})")

    finally:
        if os.path.exists(annotated_path):
            os.remove(annotated_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upload panel-5 FITS to Zooniverse as two-frame subjects")
    parser.add_argument('input', nargs='+',
                        help='panel5 FITS file(s) or directories containing them')
    parser.add_argument('--subject-set-id', type=int, default=SUBJECT_SET_ID,
                        help=f'Zooniverse subject set ID (default: {SUBJECT_SET_ID})')
    parser.add_argument('--username', default=os.environ.get('ZOONIVERSE_USERNAME'),
                        help='Zooniverse username')
    parser.add_argument('--password', default=os.environ.get('ZOONIVERSE_PASSWORD'),
                        help='Zooniverse password')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Annotated PNG render DPI (default: 300)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Render images but do not upload')
    args = parser.parse_args()

    if not args.dry_run and (not args.username or not args.password):
        parser.error("Provide --username/--password or set "
                     "ZOONIVERSE_USERNAME/ZOONIVERSE_PASSWORD env vars")

    fits_files = []
    for inp in args.input:
        if os.path.isdir(inp):
            for f in sorted(os.listdir(inp)):
                if f.endswith('_panel5.fits'):
                    fits_files.append(os.path.join(inp, f))
        elif os.path.isfile(inp):
            fits_files.append(inp)
        else:
            print(f"Warning: {inp} not found, skipping")

    if not fits_files:
        print("No panel5 FITS files found.")
        sys.exit(1)

    print(f"Found {len(fits_files)} FITS file(s) to process")

    if not args.dry_run:
        Panoptes.connect(username=args.username, password=args.password)
        subject_set = SubjectSet.find(args.subject_set_id)
        print(f"Connected. Subject set: {args.subject_set_id}")
    else:
        subject_set = None

    for fits_path in fits_files:
        try:
            upload_fits(fits_path, subject_set, dpi=args.dpi, dry_run=args.dry_run)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
