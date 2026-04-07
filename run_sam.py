"""
SAM (Segment Anything Model) runner for Euclid zeroth-order detections.

Picks one panel5.fits from Images/output, filters to confirmed zeroth-order
detections (TYPE == 'confirmed ZO', shown as blue boxes in the pipeline
panel plots), makes a cutout for each one, runs SAM with the bbox as prompt,
and saves per-detection results to Images/sam_output/:

  <stem>_zo<NNN>_cutout.fits     — input cutout, float32 science data
  <stem>_zo<NNN>_mask.fits       — binary SAM mask, uint8
  <stem>_zo<NNN>_result.jpg      — 2-panel figure: input cutout | SAM mask

Usage:
    python run_sam.py [--fits path/to/panel5.fits] [--checkpoint path/to/sam.pth]
                      [--model-type vit_b|vit_l|vit_h] [--pad 20]
                      [--max-dets N] [--output-dir Images/sam_output]
"""

import os
import sys
import argparse
import urllib.request
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# SAM checkpoint URLs (Meta's official releases)
# ---------------------------------------------------------------------------
SAM_CHECKPOINTS = {
    "vit_b": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_h_4b8939.pth",
    ),
}

zscale = ZScaleInterval()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_checkpoint(model_type: str, dest_dir: str = ".") -> str:
    url, fname = SAM_CHECKPOINTS[model_type]
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest):
        print(f"  Checkpoint already present: {dest}")
        return dest
    sizes = {"vit_b": "375 MB", "vit_l": "1.25 GB", "vit_h": "2.56 GB"}
    print(f"  Downloading SAM {model_type} checkpoint (~{sizes[model_type]}) …")
    print(f"  URL: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        sys.stdout.write(f"\r  {pct:.1f}%  ({downloaded // 1_048_576} / {total_size // 1_048_576} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()
    return dest


def to_rgb(data: np.ndarray) -> np.ndarray:
    """Convert 2-D float32 science array to uint8 H×W×3 RGB via ZScale."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros((*data.shape, 3), dtype=np.uint8)
    try:
        vmin, vmax = zscale.get_limits(finite)
    except Exception:
        vmin, vmax = np.nanpercentile(data, 1), np.nanpercentile(data, 99)
    normed = np.clip((data - vmin) / (vmax - vmin + 1e-12), 0, 1)
    normed = np.nan_to_num(normed, nan=0.0)
    gray = (normed * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def save_two_panel(cutout_rgb: np.ndarray, mask: np.ndarray,
                   score: float, bbox: tuple, det_idx: int, path: str,
                   prompt_point: tuple | None = None) -> None:
    """Save a side-by-side figure: input cutout (left) | SAM mask (right).

    prompt_point: (px, py) in cutout-local coords — drawn on the left panel
                  when the point-prompt mode was used (yellow detections).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"det#{det_idx:03d}  |  bbox {bbox}  |  SAM score {score:.3f}",
        fontsize=9,
    )

    # Left panel — input cutout (+ point marker if provided)
    axes[0].imshow(cutout_rgb, origin="upper")
    if prompt_point is not None:
        px, py = prompt_point
        axes[0].plot(px, py, marker="*", color="yellow",
                     markersize=10, markeredgecolor="black", markeredgewidth=0.5)
        axes[0].set_title("Input cutout  ★ = brightest px", fontsize=8)
    else:
        axes[0].set_title("Input cutout (ZScale)", fontsize=8)
    axes[0].axis("off")

    # Right panel — SAM mask overlaid on cutout
    axes[1].imshow(cutout_rgb, origin="upper")
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = [0.0, 0.8, 1.0, 0.55]   # cyan, 55% alpha
    axes[1].imshow(overlay, origin="upper")
    axes[1].set_title(f"SAM mask  (fill {mask.mean():.2%})", fontsize=8)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run SAM on confirmed zeroth-order detections in a panel5.fits file"
    )
    parser.add_argument("--fits", default=None,
                        help="Path to a panel5.fits file. "
                             "Defaults to the first one found in Images/output/")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SAM checkpoint .pth. Downloaded automatically if absent.")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model size (default: vit_b — fastest, ~375 MB)")
    parser.add_argument("--pad", type=int, default=20,
                        help="Pixel padding added around each detection bbox (default: 20)")
    parser.add_argument("--max-dets", type=int, default=None,
                        help="Process at most this many ZO detections (default: all)")
    parser.add_argument("--output-dir", default="Images/sam_output",
                        help="Output directory (default: Images/sam_output)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Locate panel5.fits
    # ------------------------------------------------------------------
    if args.fits:
        fits_path = args.fits
    else:
        out_dir = "Images/output"
        candidates = sorted(f for f in os.listdir(out_dir) if f.endswith("_panel5.fits"))
        if not candidates:
            sys.exit("No *_panel5.fits files found in Images/output/")
        fits_path = os.path.join(out_dir, candidates[0])

    print(f"FITS file : {fits_path}")

    # ------------------------------------------------------------------
    # 2. Load image + detections; split by type
    # ------------------------------------------------------------------
    with fits.open(fits_path) as hdul:
        image_data = hdul[2].data.astype(np.float32)   # HDU 2: 2040×2040
        det_table  = hdul[3].data                        # HDU 3: detection table

    h, w = image_data.shape
    types = np.array([t.strip() for t in det_table["TYPE"]])

    zo_rows     = det_table[types == "confirmed ZO"]
    green_rows  = det_table[types == "confident"]
    yellow_rows = det_table[types == "suspicious"]

    n_zo          = len(zo_rows)
    n_green       = len(green_rows)
    n_yellow      = len(yellow_rows)
    n_proc        = min(n_zo,     args.max_dets) if args.max_dets else n_zo
    n_green_proc  = min(n_green,  args.max_dets) if args.max_dets else n_green
    n_yellow_proc = min(n_yellow, args.max_dets) if args.max_dets else n_yellow

    print(f"Image     : {h} × {w} px")
    print(f"Detections found:")
    print(f"  confirmed ZO  (blue boxes)   : {n_zo}     → processing {n_proc}")
    print(f"  confident     (green boxes)  : {n_green}  → processing {n_green_proc}")
    print(f"  suspicious    (yellow boxes) : {n_yellow} → processing {n_yellow_proc}")

    if n_proc == 0 and n_green_proc == 0 and n_yellow_proc == 0:
        sys.exit("No detections found in this file.")

    # ------------------------------------------------------------------
    # 3. SAM checkpoint
    # ------------------------------------------------------------------
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = args.checkpoint or download_checkpoint(args.model_type, dest_dir=ckpt_dir)

    # ------------------------------------------------------------------
    # 4. Load SAM model
    # ------------------------------------------------------------------
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM {args.model_type} on {device} …")
    sam = sam_model_registry[args.model_type](checkpoint=ckpt_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    # ------------------------------------------------------------------
    # 5. Pre-compute full-image RGB (ZScale; reused for all cutouts)
    # ------------------------------------------------------------------
    full_rgb = to_rgb(image_data)

    # ------------------------------------------------------------------
    # 6. Output directories + file stem
    # ------------------------------------------------------------------
    zo_dir     = os.path.join(args.output_dir, "zo")
    green_dir  = os.path.join(args.output_dir, "green")
    yellow_dir = os.path.join(args.output_dir, "yellow")
    os.makedirs(zo_dir,     exist_ok=True)
    os.makedirs(green_dir,  exist_ok=True)
    os.makedirs(yellow_dir, exist_ok=True)
    stem = os.path.basename(fits_path).replace("_panel5.fits", "")
    pad  = args.pad

    # ------------------------------------------------------------------
    # 7. Run SAM on both categories
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 8. Helper: run SAM on one detection list and save to a subdirectory
    # ------------------------------------------------------------------
    def run_on_detections(rows, n, subdir, prefix, label, prompt_mode="box"):
        """prompt_mode: 'box' (default) or 'bright_point' (uses brightest pixel in bbox)."""
        if n == 0:
            print(f"\nNo {label} detections — skipping.")
            return
        print(f"\nRunning SAM on {n} {label} detections  [prompt={prompt_mode}] …\n")
        for i in range(n):
            row = rows[i]
            x0, y0 = int(row["X_MIN"]), int(row["Y_MIN"])
            x1, y1 = int(row["X_MAX"]), int(row["Y_MAX"])

            cy0 = max(0, y0 - pad);  cy1 = min(h, y1 + pad)
            cx0 = max(0, x0 - pad);  cx1 = min(w, x1 + pad)

            cutout_data = image_data[cy0:cy1, cx0:cx1]
            cutout_rgb  = full_rgb[cy0:cy1, cx0:cx1]

            if cutout_rgb.size == 0:
                print(f"  [{prefix}#{i:03d}] Skipping — empty cutout at ({x0},{y0})-({x1},{y1})")
                continue

            predictor.set_image(cutout_rgb)

            if prompt_mode == "bright_point":
                # Find brightest finite pixel inside the detection bbox (not the padded cutout)
                bbox_region = image_data[y0:y1, x0:x1]
                finite_mask = np.isfinite(bbox_region)
                if finite_mask.any():
                    flat_idx = np.nanargmax(np.where(finite_mask, bbox_region, -np.inf))
                    by, bx = np.unravel_index(flat_idx, bbox_region.shape)
                else:
                    # Fallback: bbox centre
                    by, bx = (y1 - y0) // 2, (x1 - x0) // 2
                # Convert to cutout-local coordinates
                pt_x = (x0 + bx) - cx0
                pt_y = (y0 + by) - cy0
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[pt_x, pt_y]], dtype=np.float32),
                    point_labels=np.array([1]),   # 1 = foreground
                    multimask_output=False,
                )
                prompt_point = (pt_x, pt_y)
            else:
                box_local = np.array([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0], dtype=np.float32)
                masks, scores, _ = predictor.predict(
                    box=box_local[None, :],
                    multimask_output=False,
                )
                prompt_point = None

            mask  = masks[0]
            score = float(scores[0])

            tag = f"{stem}_{prefix}{i:03d}"

            fits.PrimaryHDU(cutout_data.astype(np.float32)).writeto(
                os.path.join(subdir, f"{tag}_cutout.fits"), overwrite=True
            )
            fits.PrimaryHDU(mask.astype(np.uint8)).writeto(
                os.path.join(subdir, f"{tag}_mask.fits"), overwrite=True
            )
            save_two_panel(
                cutout_rgb, mask, score,
                bbox=(x0, y0, x1, y1),
                det_idx=i,
                path=os.path.join(subdir, f"{tag}_result.jpg"),
                prompt_point=prompt_point,
            )

            pt_info = f" | point=({pt_x},{pt_y})" if prompt_point else ""
            print(
                f"  {prefix}#{i:03d} | bbox=({x0},{y0})-({x1},{y1}){pt_info} | "
                f"cutout={cutout_data.shape} | score={score:.3f} | fill={mask.mean():.2%}"
            )

    # ------------------------------------------------------------------
    # 9. Run both categories
    # ------------------------------------------------------------------
    run_on_detections(zo_rows,     n_proc,        zo_dir,     "zo",     "confirmed ZO")
    run_on_detections(green_rows,  n_green_proc,  green_dir,  "green",  "confident")
    run_on_detections(yellow_rows, n_yellow_proc, yellow_dir, "yellow", "suspicious",
                      prompt_mode="bright_point")

    print(f"\nDone.")
    print(f"  {zo_dir}/     — confirmed ZO  (blue boxes)")
    print(f"  {green_dir}/  — confident     (green boxes)")
    print(f"  {yellow_dir}/ — suspicious    (yellow boxes)")
    print(f"  Each subfolder: *_cutout.fits, *_mask.fits, *_result.jpg")


if __name__ == "__main__":
    main()
