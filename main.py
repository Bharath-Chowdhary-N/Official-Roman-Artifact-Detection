"""
Euclid NISP SCIFRM - 5-Panel Pipeline

For each detector (DETxx.SCI and its DETxx.DQ quality map) this pipeline
produces five views side by side:

    Panel 1  Raw science image displayed with ZScale stretch
    Panel 2  Continuum-subtracted image where 2-D spectra are removed
             and point sources are isolated
    Panel 3  Continuum-subtracted image with DQ-flagged hot pixels set
             to NaN so they appear black
    Panel 4  Continuum-subtracted image with hot pixels replaced by
             locally-estimated background noise
    Panel 5  Object detections overlaid on panel 4 - confident detections
             shown in green, suspicious ones in yellow

Outputs written to output/ by default:
    - 5-panel JPEG or PNG at 150 DPI
    - Panel-5-only JPEG or PNG at 300 DPI
    - Panel-5-only FITS file preserving the inpainted continuum-subtracted data
    - A plain-text summary file

Usage:
    python full_pipeline_5panel.py <fits_file> [options]
    python full_pipeline_5panel.py --help
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.ndimage import median_filter, uniform_filter, rotate
from skimage import morphology, measure
from datetime import datetime
import json
import pandas as pd
import gelsa

G = gelsa.Gelsa("calib/gelsa_config.json")
ESC_PATH = "Euclid Extracted Spectra Data"
ESC_ALL_PATH = os.path.join(ESC_PATH, "bigdigest 5614B8BE632B859E72E37665FB9914E3.csv")

ZO_MIN_DST_THRESHOLD = 0.75
ZO_H_SCALE = 5.0
ZO_W_SCALE = 20.0
ZO_MIN_BRIGHTNESS_MAG = 19
ZO_ALGORITHM_VERSION = 2.0
ZO_ALGORITHM_PARAMS = {
    "ZO_MIN_DST_THRESHOLD": ZO_MIN_DST_THRESHOLD,
    "ZO_H_SCALE": ZO_H_SCALE,
    "ZO_W_SCALE": ZO_W_SCALE,
    "ZO_MIN_BRIGHTNESS_MAG": ZO_MIN_BRIGHTNESS_MAG
}

warnings.filterwarnings('ignore')
zscale = ZScaleInterval()

class DetIndex:
    def __init__(self, thing):
        if type(thing) is str:
            self.code = thing
            self.idx = (int(self.code[1])-1)*4 + (int(self.code[0])-1)
        elif type(thing) is int:
            self.idx = thing
            self.code = f"{self.idx%4 + 1}{self.idx//4 + 1}"
        else:
            raise Exception("What'd you give me?")
DetIndex.ALL_DETS = [DetIndex(i) for i in range(16)]

def get_grism_angle(header):
    return header["GWA_TILT"] + float(header["GWA_POS"][-3:])

def center_crop(img, crop_dims):
    """
    Returns a center-cropped image (NumPy array).

    Args:
        img (np.ndarray): The input image as a NumPy array (H, W, C or H, W).
        crop_dims (tuple): The dimensions to crop to (crop_height, crop_width).

    Returns:
        np.ndarray: The center-cropped image.
    """
    img_height, img_width = img.shape[:2]
    crop_height, crop_width = crop_dims
    start_height = (img_height - crop_height) // 2
    start_width = (img_width - crop_width) // 2
    end_height = start_height + crop_height
    end_width = start_width + crop_width
    return img[start_height:end_height, start_width:end_width, ...]

def calc_cont_single(image, grism_angle, kernel_size=(1, 41)):
    angle = float(grism_angle)
    rot_angle = angle - 180.0 if angle > 90.0 else angle
    rotating = not np.isclose(rot_angle, 0.0)
    # Note the sign here! Because we use origin="lower", the sign is flipped.
    if rotating:
        A = image.copy()
        nan_mask = np.isnan(image)
        A[nan_mask] = 0.0
        A = rotate(A, angle=rot_angle, reshape=True)
    A = median_filter(A if rotating else image, size=kernel_size, mode='nearest')
    if rotating:
        A = rotate(A, angle=-rot_angle, reshape=True)
        A = center_crop(A, image.shape[:2])
        A[nan_mask] = np.nan
    return A

def cont_subtract_single(image, grism_angle, kernel_size=(1, 41)):
    """
    Subtract the spectral continuum from one detector image.

    For grism angles of 0 or 180 degrees a plain horizontal median filter is
    applied. For the small tilts used by Euclid (around plus or minus 4 degrees)
    the image is first rotated to align spectra with the pixel grid, filtered,
    then rotated back. Returns the continuum-subtracted image.
    """
    return image - calc_cont_single(image, grism_angle, kernel_size)


def build_hot_pixel_mask(hdul, dq_name, hot_pixel_bits=0xFFFFFFFF):
    """
    Build a boolean mask of hot and bad pixels from the DETxx.DQ extension.
    Returns None if the named extension is not present in the file.
    """
    all_names = [h.name for h in hdul]
    if dq_name not in all_names:
        print(f"  WARNING: '{dq_name}' not found, no hot pixels flagged")
        return None

    dq_raw   = hdul[dq_name].data
    dq       = dq_raw.astype(np.int64)
    dq       = np.where(dq < 0, dq + 2**32, dq).astype(np.uint32)
    hot_mask = (dq & hot_pixel_bits).astype(bool)

    n_hot = hot_mask.sum()
    print(f"  DQ hot pixels: {n_hot:,}  ({100 * n_hot / hot_mask.size:.3f}%)")

    unique_vals, counts = np.unique(dq[dq > 0], return_counts=True)
    if len(unique_vals):
        print("  Non-zero DQ values (top 10):")
        for v, c in zip(unique_vals[:10], counts[:10]):
            bits = [b for b in range(32) if v & (1 << b)]
            print(f"    DQ={int(v):>10}  count={c:>8}  bits={bits}")

    return hot_mask


def inpaint_hot_pixels(image, hot_mask, box_size=64, min_good_fraction=0.3):
    """
    Replace flagged hot pixel locations with locally estimated background plus
    Gaussian noise scaled to the local variance. Falls back to global statistics
    when a pixel's neighbourhood does not contain enough good pixels.
    """
    img  = image.copy().astype(np.float64)
    good = ~hot_mask & np.isfinite(image)

    good_vals    = img[good]
    global_bg    = float(np.median(good_vals)) if good_vals.size else 0.0
    global_mad   = float(np.median(np.abs(good_vals - global_bg))) if good_vals.size else 1.0
    global_sigma = max(global_mad * 1.4826, 1e-6)
    print(f"    Inpaint: global bg={global_bg:.4g}  sigma={global_sigma:.4g}")

    gf    = good.astype(np.float64)
    clean = np.where(good, img, global_bg)

    cnt   = uniform_filter(gf,             size=box_size, mode='reflect')
    bg    = uniform_filter(clean * gf,     size=box_size, mode='reflect')
    sq    = uniform_filter(clean**2 * gf,  size=box_size, mode='reflect')

    safe  = cnt > 0
    bg    = np.where(safe, bg / (cnt + 1e-30), global_bg)
    var   = np.where(safe, sq / (cnt + 1e-30) - bg**2, global_sigma**2)
    sigma = np.sqrt(np.abs(var))

    min_cnt   = min_good_fraction * (box_size ** 2)
    use_local = cnt >= min_cnt
    bg        = np.where(use_local, bg,    global_bg)
    sigma     = np.where(use_local, sigma, global_sigma)

    rng = np.random.default_rng(42)
    img[hot_mask] = (
        bg[hot_mask] + sigma[hot_mask] * rng.normal(0.0, 1.0, img.shape)[hot_mask]
    )

    return img

#############################################################################
def zscale_01(img, nan_mask=None):
    """
    Apply ZScale stretch and normalise the result to the range 0 to 1.
    If nan_mask is provided those pixels are set to NaN so they render black
    when displayed with a gray colormap that has bad-value colour set to black.
    """
    arr    = img.copy().astype(np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    vmin, vmax = zscale.get_limits(finite)
    out = np.clip((arr - vmin) / (vmax - vmin + 1e-30), 0, 1)
    if nan_mask is not None:
        out[nan_mask] = np.nan
    return out


def detect_all_objects(image, intensity_threshold=0.05, min_area=15,
                       max_area=100000, max_objects=1000, **kwargs):
    """
    Detect sources in a normalised image using connected-component labelling.
    Pixels above intensity_threshold are gathered into a binary mask, small
    holes and specks are cleaned up, then regions are measured and returned
    sorted by priority (mean intensity times area).
    """
    print(f"  Creating intensity mask (threshold: {intensity_threshold})...")
    intensity_mask = image > intensity_threshold
    total_pixels   = np.sum(intensity_mask)
    print(f"  Found {total_pixels:,} pixels above threshold")

    print("  Cleaning mask...")
    cleaned_mask = morphology.remove_small_objects(intensity_mask, min_size=min_area // 2)
    cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=min_area // 4)

    print("  Labeling connected objects...")
    labeled  = measure.label(cleaned_mask)
    regions  = measure.regionprops(labeled, intensity_image=image)
    print(f"  Found {len(regions)} connected objects")

    object_list = []
    for region in regions:
        if region.area < min_area or region.area > max_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        obj_info = {
            'bbox'          : [minc, minr, maxc, maxr],
            'centroid'      : [region.centroid[1], region.centroid[0]],
            'area'          : region.area,
            'mean_intensity': region.mean_intensity,
            'max_intensity' : region.max_intensity,
            'width'         : maxc - minc,
            'height'        : maxr - minr,
            'label'         : region.label,
        }
        obj_info['aspect_ratio'] = (
            obj_info['width'] / obj_info['height']
            if obj_info['height'] > 0 else float('inf')
        )
        bbox_area             = obj_info['width'] * obj_info['height']
        obj_info['fill_ratio'] = obj_info['area'] / bbox_area if bbox_area > 0 else 0.0
        obj_info['priority']  = region.mean_intensity * region.area
        obj_info['pixels']    = labeled == region.label
        object_list.append(obj_info)

    object_list.sort(key=lambda x: x['priority'], reverse=True)
    object_list = object_list[:max_objects]
    print(f"  Selected {len(object_list)} objects")

    all_object_pixels = np.zeros_like(image, dtype=bool)
    for obj_info in object_list:
        all_object_pixels |= obj_info['pixels']

    return all_object_pixels, object_list, intensity_mask


def merge_objects_fast(object_list,
                       overlap_threshold=0.1,
                       proximity_threshold=30,
                       aspect_ratio_tolerance=2.0,
                       horizontal_overlap_threshold=0.3):
    """
    Merge nearby detections using a Union-Find approach. Objects are merged when
    they overlap significantly in both axes or when their bounding boxes are
    within proximity_threshold pixels of each other. Objects with very different
    aspect ratios are not merged.
    """
    if not object_list:
        return []

    print(f"\n  Starting fast object merging...")
    print(f"    Initial objects: {len(object_list)}")

    n      = len(object_list)
    parent = list(range(n))
    rank   = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx]  += 1

    bboxes        = np.array([obj['bbox']         for obj in object_list])
    centroids     = np.array([obj['centroid']      for obj in object_list])
    aspect_ratios = np.array([obj['aspect_ratio']  for obj in object_list])
    sort_idx      = np.argsort(bboxes[:, 0])

    merge_count = rejected_aspect = rejected_h = 0

    for i in range(n):
        idx_i = sort_idx[i]
        minc_i, minr_i, maxc_i, maxr_i = bboxes[idx_i]
        width_i, height_i = maxc_i - minc_i, maxr_i - minr_i
        centroid_i, aspect_i = centroids[idx_i], aspect_ratios[idx_i]
        max_search = proximity_threshold + width_i

        for j in range(i + 1, n):
            idx_j = sort_idx[j]
            minc_j, minr_j, maxc_j, maxr_j = bboxes[idx_j]
            width_j, height_j = maxc_j - minc_j, maxr_j - minr_j

            if minc_j > maxc_i + max_search:
                break

            centroid_j, aspect_j = centroids[idx_j], aspect_ratios[idx_j]

            if aspect_i > 1e-6 and aspect_j > 1e-6:
                ar = max(aspect_i, aspect_j) / min(aspect_i, aspect_j)
                if ar > aspect_ratio_tolerance:
                    rejected_aspect += 1
                    continue

            dx    = abs(centroid_i[0] - centroid_j[0])
            dy    = abs(centroid_i[1] - centroid_j[1])
            max_w = max(width_i, width_j)
            max_h = max(height_i, height_j)
            if dx > max_w + proximity_threshold or dy > max_h + proximity_threshold:
                continue

            ic_min = max(minc_i, minc_j)
            ic_max = min(maxc_i, maxc_j)
            ir_min = max(minr_i, minr_j)
            ir_max = min(maxr_i, maxr_j)
            has_overlap = ic_max > ic_min and ir_max > ir_min

            overlap_merge = False
            if has_overlap:
                iw    = ic_max - ic_min
                ih    = ir_max - ir_min
                min_w = min(width_i, width_j)
                min_h = min(height_i, height_j)
                hor   = iw / min_w if min_w > 0 else 0
                ver   = ih / min_h if min_h > 0 else 0
                overlap_merge = hor >= horizontal_overlap_threshold and ver >= overlap_threshold
                if not overlap_merge:
                    rejected_h += 1

            h_dist = max(0, minc_j - maxc_i) if maxc_i < minc_j else max(0, minc_i - maxc_j)
            v_dist = max(0, minr_j - maxr_i) if maxr_i < minr_j else max(0, minr_i - maxr_j)
            proximity_merge = np.sqrt(h_dist**2 + v_dist**2) <= proximity_threshold

            if overlap_merge or proximity_merge:
                union(idx_i, idx_j)
                merge_count += 1

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(object_list[i])

    print(f"    Final groups: {len(groups)}  (merges: {merge_count})")
    return [_merge_group(grp, gid) for gid, grp in enumerate(groups.values())]


def _merge_group(group, group_id):
    minc = min(o['bbox'][0] for o in group)
    minr = min(o['bbox'][1] for o in group)
    maxc = max(o['bbox'][2] for o in group)
    maxr = max(o['bbox'][3] for o in group)
    total_area = sum(o['area'] for o in group) or 1
    cx     = sum(o['centroid'][0] * o['area'] for o in group) / total_area
    cy     = sum(o['centroid'][1] * o['area'] for o in group) / total_area
    mean_i = sum(o['mean_intensity'] * o['area'] for o in group) / total_area
    max_i  = max(o['max_intensity'] for o in group)
    w, h   = maxc - minc, maxr - minr
    bbox_a = w * h
    merged_px = np.zeros_like(group[0]['pixels'], dtype=bool)
    for o in group:
        merged_px |= o['pixels']
    return {
        'bbox'          : [minc, minr, maxc, maxr],
        'centroid'      : [cx, cy],
        'area'          : total_area,
        'mean_intensity': mean_i,
        'max_intensity' : max_i,
        'width'         : w,
        'height'        : h,
        'aspect_ratio'  : w / h if h > 0 else float('inf'),
        'fill_ratio'    : total_area / bbox_a if bbox_a > 0 else 0.0,
        'priority'      : mean_i * total_area,
        'label'         : group_id,
        'pixels'        : merged_px,
        'fragment_count': len(group),
    }


def filter_horizontal_objects(object_list, min_aspect_ratio=3.0, min_width=50,
                               protected_labels=None):
    """
    Separate very elongated objects (likely residual spectra) from compact ones.
    Objects whose labels appear in protected_labels are kept regardless of shape.
    Returns (compact_objects, horizontal_objects).
    """
    if protected_labels is None:
        protected_labels = set()
    filtered, horizontal = [], []
    for obj in object_list:
        is_h = obj['aspect_ratio'] >= min_aspect_ratio and obj['width'] >= min_width
        if is_h and obj['label'] not in protected_labels:
            horizontal.append(obj)
        else:
            filtered.append(obj)
    return filtered, horizontal


def save_panel5_fits(
    cs_inpainted,
    confident_objects,
    suspicious_objects,
    paired_objects,
    sci_header,
    n_hot,
    output_path,
    preprocessing_params,
    detection_params,
    merge_params,
    filter_params,
    zo_table
):
    """
    Save the inpainted continuum-subtracted image (the data underlying panel 5)
    to a FITS file. Detection bounding boxes are stored as a binary table
    extension so they can be read back later.
    """

    fits.HDUList([fits.PrimaryHDU()]).writeto(output_path, overwrite=True)
    hdul_out = fits.open(output_path, mode="update")
    print(f"  Writing to FITS: {output_path}")

    detCode = os.path.splitext(sci_header["EXTNAME"])[0]
    now = str(datetime.now())

    # Write the ZO table
    if "ZOTABLE" in hdul_out:
        print("Already has ZO table")
    else:
        print("Writing ZO table")
        hdul_out.append(fits.BinTableHDU.from_columns(fits.ColDefs([
            fits.Column(name='X',        format='D', array = zo_table["ZO x"]  ),
            fits.Column(name='Y',        format='D', array = zo_table["ZO y"]  ),
            fits.Column(name='Detector', format='B', array = zo_table["ZO Det"]),
        ]), name = "ZOTABLE"))

    preprocessedIHDU = fits.ImageHDU(
        data = cs_inpainted.astype(np.float32),
        name = f"{detCode}.Preprocessed",
        header = sci_header
    )
    preprocessedIHDU.header['N_HOT']    = (n_hot, 'Number of DQ-flagged hot pixels')
    preprocessedIHDU.header['HISTORY']  = 'Continuum-subtracted inpainted image (panel 5 data)'
    preprocessedIHDU.header['PREP_ALG'] = json.dumps({
        "version": 1.0,
        "date": now,
        "params": preprocessing_params
    })

    if preprocessedIHDU.name in hdul_out:
        print(f"Overwriting existing {preprocessedIHDU.name}")
        hdul_out[preprocessedIHDU.name] = preprocessedIHDU
    else:
        hdul_out.append(preprocessedIHDU)

    all_objects = (
        [(o, 'confident')       for o in confident_objects  ] +
        [(o, 'suspicious')      for o in suspicious_objects ] + 
        [(o, 'confirmed ZO')    for o in paired_objects     ]
    )

    table_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs([
            fits.Column(name='X_MIN',    format='J',     array=np.array([o['bbox'][0]       for o, _ in all_objects])),
            fits.Column(name='Y_MIN',    format='J',     array=np.array([o['bbox'][1]       for o, _ in all_objects])),
            fits.Column(name='X_MAX',    format='J',     array=np.array([o['bbox'][2]       for o, _ in all_objects])),
            fits.Column(name='Y_MAX',    format='J',     array=np.array([o['bbox'][3]       for o, _ in all_objects])),
            fits.Column(name='CENT_X',   format='E',     array=np.array([o['centroid'][0]   for o, _ in all_objects])),
            fits.Column(name='CENT_Y',   format='E',     array=np.array([o['centroid'][1]   for o, _ in all_objects])),
            fits.Column(name='AREA',     format='J',     array=np.array([o['area']          for o, _ in all_objects])),
            fits.Column(name='FILL',     format='E',     array=np.array([o['fill_ratio']    for o, _ in all_objects])),
            fits.Column(name='ASPECT',   format='E',     array=np.array([o['aspect_ratio']  for o, _ in all_objects])),
            fits.Column(name='TYPE',     format='12A',   array=np.array([t                  for _, t in all_objects]))
        ]),
        name = f"{detCode}.Detections",
        header = sci_header
    )
    table_hdu.header['N_CONF']   = (len(confident_objects),  'Confident detections')
    table_hdu.header['N_SUSP']   = (len(suspicious_objects), 'Suspicious detections')
    table_hdu.header['HISTORY']  = 'Intensity Segmentation'
    table_hdu.header['SEG_ALG'] = json.dumps({
        "version": 2.0,
        "date": now,
        "params": {
            "detection_params": detection_params,
            "merge_params": merge_params,
            "filter_params": filter_params
        }
    })
    table_hdu.header['ZOF_ALG'] = json.dumps({
        "version": ZO_ALGORITHM_VERSION,
        "date": now,
        "params": ZO_ALGORITHM_PARAMS
    })

    if table_hdu.name in hdul_out:
        print(f"Overwriting existing {table_hdu.name}")
        hdul_out[table_hdu.name] = table_hdu
    else:
        hdul_out.append(table_hdu)
    hdul_out.close(output_verify='silentfix')


def process_detector(hdul, det_idx, output_dir, base_name,
                     kernel_size, inpaint_box, hot_pixel_bits,
                     image_format, jpeg_quality,
                     detection_params, merge_params, filter_params,
                     zo_table,
                     save_5panel=True, save_panel5_standalone=True,
                     save_panel5_fits_flag=True):

    sci_name = f"DET{det_idx.code}"
    dq_name  = f"{sci_name}.DQ"
    sci_name  = f"{sci_name}.SCI"

    print(f"\n{'=' * 60}")
    print(f"Detector {det_idx.idx + 1}/{len([x for x in hdul if x.name.endswith('.SCI')])}  ->  {sci_name}")
    print(f"{'=' * 60}")

    sci_data    = hdul[sci_name].data.astype(np.float64)
    sci_header  = hdul[sci_name].header
    grism_angle = get_grism_angle(hdul[0].header)

    print(f"  Continuum subtraction (grism={grism_angle:.1f} deg, kernel={kernel_size})...")
    cs_image = cont_subtract_single(sci_data, grism_angle, kernel_size)

    hot_mask = build_hot_pixel_mask(hdul, dq_name, hot_pixel_bits)
    if hot_mask is None:
        hot_mask = np.zeros(sci_data.shape, dtype=bool)
    n_hot = hot_mask.sum()

    print("  Inpainting hot pixels in continuum-subtracted image...")
    cs_inpainted = inpaint_hot_pixels(cs_image, hot_mask, box_size=inpaint_box)

    print("\n  Object detection (panel 5)...")

    raw_det   = cs_inpainted.copy()
    finite_px = raw_det[np.isfinite(raw_det)]

    neg_px = finite_px[finite_px < 0]
    if neg_px.size > 100:
        bg_median = float(np.median(finite_px))
        sigma_est = float(np.std(neg_px)) * np.sqrt(2)
    else:
        bg_median = float(np.median(finite_px))
        sigma_est = float(np.median(np.abs(finite_px - bg_median))) * 1.4826

    sigma_est = max(sigma_est, 1e-9)
    n_sigma   = detection_params.get('detection_sigma', 3.0) ####################################################################################################
    flux_threshold = bg_median + n_sigma * sigma_est

    print(f"  Background median : {bg_median:.4g}")
    print(f"  Noise sigma       : {sigma_est:.4g}")
    print(f"  Flux threshold    : {flux_threshold:.4g}  ({n_sigma:.1f} sigma above bg)")

    thr_norm  = detection_params.get('intensity_threshold', 0.05)
    scale     = (flux_threshold - bg_median) / thr_norm #########################################################################################################
    det_image = np.clip((raw_det - bg_median) / (scale + 1e-30), 0, 1)

    _, initial_objects, intensity_mask = detect_all_objects(det_image, **detection_params)
    merged_objects = merge_objects_fast(initial_objects, **merge_params)

    fill_threshold  = filter_params['fill_ratio_threshold']
    suspicious_lbl  = {obj['label'] for obj in merged_objects if obj['fill_ratio'] < fill_threshold}
    filtered_objects, horizontal_objects = filter_horizontal_objects(
        merged_objects,
        min_aspect_ratio=filter_params['min_aspect_ratio'],
        min_width=filter_params['min_width'],
        protected_labels=suspicious_lbl,
    )

    confident_objects  = [o for o in filtered_objects if o['fill_ratio'] >= fill_threshold]
    suspicious_objects = [o for o in filtered_objects if o['fill_ratio'] <  fill_threshold]
    print(f"  Confident (green)\t: {len(confident_objects)}")
    print(f"  Suspicious (yellow)\t: {len(suspicious_objects)}")

    # Segregate possible ZOs out based on distance
    zx = zo_table[zo_table["ZO Det"] == det_idx.idx]["ZO x"]
    zy = zo_table[zo_table["ZO Det"] == det_idx.idx]["ZO y"]
    zpos = np.column_stack((zx, zy))
    paired_objects = []
    tilt = np.deg2rad(grism_angle)
    A_inv = np.linalg.inv(np.array([
        [ZO_W_SCALE * np.cos(tilt), -ZO_H_SCALE * np.sin(tilt)],
        [ZO_W_SCALE * np.sin(tilt),  ZO_H_SCALE * np.cos(tilt)]
    ]))
    for o in confident_objects:
        dsts = zpos.copy()
        region_center = np.array(o['centroid'])
        dsts -= region_center
        dsts = np.linalg.norm(np.transpose(np.inner(A_inv, dsts)), axis=1)
        if np.min(dsts) < ZO_MIN_DST_THRESHOLD:
            paired_objects.append(o)
    confident_objects = [o for o in confident_objects if o not in paired_objects]
    print(f"  Confirmed ZOs (blue)\t: {len(paired_objects)}, this leaves {len(confident_objects)} confident (green) objects remaining")

    disp_raw          = zscale_01(sci_data)
    disp_cs           = zscale_01(cs_image)
    disp_cs_hot_nan   = zscale_01(cs_image,     nan_mask=hot_mask)
    disp_cs_inpainted = zscale_01(cs_inpainted)

    cmap = plt.cm.gray.copy()
    cmap.set_bad('black')

    fname_stem = f"{base_name}_{sci_name.replace('.', '_')}"

    if save_5panel:
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(
            f"{base_name}  |  {sci_name}  |  grism = {grism_angle:.1f} deg\n" +
            f"DQ hot pixels: {n_hot:,}  ({100 * n_hot / hot_mask.size:.3f}%)  " +
            f"|  Detected: {len(confident_objects)} confident  " +
            f"{len(suspicious_objects)} suspicious" +
            f"{len(paired_objects)} confirms ZOs",
            fontsize=10,
        )

        axes[0].imshow(disp_raw,          origin='lower', cmap=cmap, vmin=0, vmax=1)
        axes[0].set_title("1. Raw science image\n(ZScale)", fontsize=9)

        axes[1].imshow(disp_cs,           origin='lower', cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title("2. Continuum-subtracted\n(2D spectra removed)", fontsize=9)

        axes[2].imshow(disp_cs_hot_nan,   origin='lower', cmap=cmap, vmin=0, vmax=1)
        axes[2].set_title(
            f"3. Hot pixels removed\n(shown black, {n_hot:,} px)", fontsize=9)

        axes[3].imshow(disp_cs_inpainted, origin='lower', cmap=cmap, vmin=0, vmax=1)
        axes[3].set_title("4. Hot pixels replaced\nwith background noise", fontsize=9)

        axes[4].imshow(disp_cs_inpainted, origin='lower', cmap=cmap, vmin=0, vmax=1)
        axes[4].set_title(
            f"5. Object detection\n" +
            f"green={len(confident_objects)} confident  " +
            f"yellow={len(suspicious_objects)} suspicious" +
            f"blue={len(paired_objects)} confirmed ZO",
            fontsize=9,
        )
        for obj in confident_objects:
            bc = obj['bbox']
            rect = patches.Rectangle(
                (bc[0], bc[1]), bc[2] - bc[0], bc[3] - bc[1],
                linewidth=1, edgecolor='lime', facecolor='none', alpha=0.9,
            )
            axes[4].add_patch(rect)
        for obj in suspicious_objects:
            bc = obj['bbox']
            rect = patches.Rectangle(
                (bc[0], bc[1]), bc[2] - bc[0], bc[3] - bc[1],
                linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.9,
            )
            axes[4].add_patch(rect)
        for obj in paired_objects:
            bc = obj['bbox']
            rect = patches.Rectangle(
                (bc[0], bc[1]), bc[2] - bc[0], bc[3] - bc[1],
                linewidth=1, edgecolor='blue', facecolor='none', alpha=0.9,
            )
            axes[4].add_patch(rect)
        axes[4].scatter(zx, zy, marker="s", facecolors='none', edgecolors='r', linewidths=1)
        for ax in axes:
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")

        plt.tight_layout()

        ext = 'jpg' if image_format.lower() in ('jpeg', 'jpg') else 'png'
        if ext == 'jpg':
            pkl = {'format': 'jpeg', 'quality': jpeg_quality, 'optimize': True}
        else:
            pkl = {'format': 'png', 'optimize': True}

        panel_path = os.path.join(output_dir, f"{fname_stem}_5panel.{ext}")
        plt.savefig(panel_path, dpi=150, bbox_inches='tight', pil_kwargs=pkl)
        plt.close()
        print(f"  Saved 5-panel: {panel_path}")

    if save_panel5_standalone:
        fig5, ax5 = plt.subplots(1, 1, figsize=(10, 10))
        ax5.imshow(disp_cs_inpainted, origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax5.set_title(
            f"{sci_name}  |  grism = {grism_angle:.1f} deg\n"
            f"Detections: {len(confident_objects)} confident (green), "
            f"{len(suspicious_objects)} suspicious (yellow)",
            fontsize=10,
        )
        ax5.set_xlabel("X (px)")
        ax5.set_ylabel("Y (px)")
        for obj in confident_objects:
            bc = obj['bbox']
            rect = patches.Rectangle(
                (bc[0], bc[1]), bc[2] - bc[0], bc[3] - bc[1],
                linewidth=1, edgecolor='lime', facecolor='none', alpha=0.9,
            )
            ax5.add_patch(rect)
        for obj in suspicious_objects:
            bc = obj['bbox']
            rect = patches.Rectangle(
                (bc[0], bc[1]), bc[2] - bc[0], bc[3] - bc[1],
                linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.9,
            )
            ax5.add_patch(rect)
        for obj in paired_objects:
            bc = obj['bbox']
            rect = patches.Rectangle(
                (bc[0], bc[1]), bc[2] - bc[0], bc[3] - bc[1],
                linewidth=1, edgecolor='blue', facecolor='none', alpha=0.9,
            )
            ax5.add_patch(rect)
        ax5.scatter(zx, zy, marker="s", facecolors='none', edgecolors='r', linewidths=1)
        plt.tight_layout()

        ext = 'jpg' if image_format.lower() in ('jpeg', 'jpg') else 'png'
        if ext == 'jpg':
            pkl = {'format': 'jpeg', 'quality': jpeg_quality, 'optimize': True}
        else:
            pkl = {'format': 'png', 'optimize': True}

        p5_path = os.path.join(output_dir, f"{fname_stem}_panel5.{ext}")
        plt.savefig(p5_path, dpi=300, bbox_inches='tight', pil_kwargs=pkl)
        plt.close()
        print(f"  Saved panel 5 standalone (300 DPI): {p5_path}")

        # Clean version: no axes, no title, no boxes — for Zooniverse upload
        fig_clean, ax_clean = plt.subplots(1, 1, figsize=(10, 10))
        ax_clean.imshow(disp_cs_inpainted, origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax_clean.axis('off')
        p5_clean_path = os.path.join(output_dir, f"{fname_stem}_panel5_clean.{ext}")
        plt.savefig(p5_clean_path, dpi=300, bbox_inches='tight', pad_inches=0,
                    pil_kwargs=pkl)
        plt.close()
        print(f"  Saved panel 5 clean (300 DPI): {p5_clean_path}")

    if save_panel5_fits_flag:
        fits_path_out = os.path.join(output_dir, f"{fname_stem}_panel5.fits")
        save_panel5_fits(
            cs_inpainted,
            confident_objects,
            suspicious_objects,
            paired_objects,
            sci_header,
            n_hot,
            fits_path_out,
            { "inpaint_box": inpaint_box, "kernel_size": kernel_size, "hot_pixel_bits": hot_pixel_bits },
            detection_params,
            merge_params,
            filter_params,
            zo_table
        )

    # Build per-detection rows for CSV output
    detection_rows = []
    for obj_type, obj_list in [('confident', confident_objects),
                                ('suspicious', suspicious_objects),
                                ('confirmed ZO', paired_objects)]:
        for obj in obj_list:
            detection_rows.append({
                'detector'      : sci_name,
                'type'          : obj_type,
                'x_min'         : obj['bbox'][0],
                'y_min'         : obj['bbox'][1],
                'x_max'         : obj['bbox'][2],
                'y_max'         : obj['bbox'][3],
                'centroid_x'    : obj['centroid'][0],
                'centroid_y'    : obj['centroid'][1],
                'width'         : obj['width'],
                'height'        : obj['height'],
                'area'          : obj['area'],
                'fill_ratio'    : obj['fill_ratio'],
                'aspect_ratio'  : obj['aspect_ratio'],
                'mean_intensity': obj['mean_intensity'],
                'max_intensity' : obj['max_intensity'],
                'priority'      : obj['priority'],
                'label'         : obj['label'],
                'fragment_count': obj.get('fragment_count', 1),
            })

    return {
        'sci_name'      : sci_name,
        'grism_angle'   : grism_angle,
        'hot_px'        : int(n_hot),
        'hot_pct'       : round(100 * n_hot / hot_mask.size, 3),
        'initial_objects': len(initial_objects),
        'merged_objects' : len(merged_objects),
        'horiz_removed'  : len(horizontal_objects),
        'confident'      : len(confident_objects),
        'suspicious'     : len(suspicious_objects),
        'confirmed ZOs'  : len(paired_objects),
        'detection_rows' : detection_rows,
    }


def process_fits_with_full_pipeline(
    fits_path,
    zo_table,
    output_dir=None,
    kernel_size=(1, 41),
    inpaint_box=64,
    hot_pixel_bits=0xFFFFFFFF,
    specific_det_index=None,
    image_format='jpeg',
    jpeg_quality=85,
    detection_params=None,
    merge_params=None,
    filter_params=None,
    save_5panel=True,
    save_panel5_standalone=True,
    save_panel5_fits_flag=True,
):
    print(f"\n{'=' * 60}")
    print("Euclid NISP - 5-Panel Pipeline")
    print("  1:raw  2:cont-sub  3:hot-removed  4:hot-noise  5:detections")
    print(f"{'=' * 60}")
    print(f"File: {fits_path}")

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(fits_path) or ".", "output"
        )
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(fits_path))[0]

    if detection_params is None:
        detection_params = {
            'intensity_threshold': 0.05,
            'detection_sigma'    : 3.0,
            'min_area'           : 15,
            'max_area'           : 100000,
            'max_objects'        : 1000,
        }
    if merge_params is None:
        merge_params = {
            'overlap_threshold'           : 0.1,
            'horizontal_overlap_threshold': 0.3,
            'proximity_threshold'         : 30,
            'aspect_ratio_tolerance'      : 2.0,
        }
    if filter_params is None:
        filter_params = {
            'min_aspect_ratio'    : 3.0,
            'min_width'           : 50,
            'fill_ratio_threshold': 0.20,
        }

    gelsa_frame = G.load_spec_frame(fits_path)

    with fits.open(fits_path, memmap=False) as hdul:
        hdul.info()
        sci_hdus    = [DetIndex(h.name[3:-4]) for h in hdul if h.name.endswith('.SCI')]
        n_det       = len(sci_hdus)
        det_indices = [DetIndex(specific_det_index)] if specific_det_index is not None else sci_hdus
        print(f"\nFound {n_det} SCI detectors.")
        all_results = []
        zo_x, zo_y, zo_det = gelsa_frame.radec_to_pixel(
            zo_table['RIGHT_ASCENSION'],
            zo_table['DECLINATION'],
            15000*np.ones(len(zo_table)),
            dispersion_order = 0
        )
        on_frame_mask               = zo_det >= 0
        on_frame_zo_table           = zo_table.iloc[on_frame_mask].copy()
        on_frame_zo_table["ZO x"]   = zo_x[on_frame_mask]
        on_frame_zo_table["ZO y"]   = zo_y[on_frame_mask]
        on_frame_zo_table["ZO Det"] = zo_det[on_frame_mask]
        for det_idx in det_indices:
            result = process_detector(
                hdul, det_idx, output_dir, base_name,
                kernel_size, inpaint_box, hot_pixel_bits,
                image_format, jpeg_quality,
                detection_params, merge_params, filter_params,
                on_frame_zo_table,
                save_5panel=save_5panel,
                save_panel5_standalone=save_panel5_standalone,
                save_panel5_fits_flag=save_panel5_fits_flag,
            )
            if result:
                all_results.append(result)

    # Save all detection boxes to a single CSV for this FITS file
    all_detection_rows = []
    for r in all_results:
        for row in r.get('detection_rows', []):
            row_with_meta = {'fits_file': base_name}
            row_with_meta.update(row)
            all_detection_rows.append(row_with_meta)
    if all_detection_rows:
        csv_path = os.path.join(output_dir, f"{base_name}_detections.csv")
        pd.DataFrame(all_detection_rows).to_csv(csv_path, index=False)
        print(f"  Saved detections CSV: {csv_path}")

    summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"File          : {fits_path}\n")
        f.write(f"Kernel        : {kernel_size}\n")
        f.write(f"Inpaint box   : {inpaint_box} px\n")
        f.write(f"Hot px bits   : 0x{hot_pixel_bits:08X}\n")
        f.write(f"Detection params: {detection_params}\n")
        f.write(f"Merge params    : {merge_params}\n")
        f.write(f"Filter params   : {filter_params}\n")
        f.write(f"ZO Filtering params   : {ZO_ALGORITHM_PARAMS}\n")
        f.write(f"{'_' * 90}\n")
        hdr = (
            f"{'Detector':<20}\t{'Grism':>7}deg\t{'Hot px':>10}\t{'%':>8}\t{'Confident':>6}\t{'Susp':>6}\t{'Confirmed':>6}\n"
        )
        f.write(hdr)
        f.write(f"{'_' * 90}\n")
        for r in all_results:
            f.write(
                f"{r['sci_name']:<20}\t{r['grism_angle']:>7.1f}\t\t{r['hot_px']:>10,}\t{r['hot_pct']:>8.3f}%\t{r['confident']:>6}\t\t{r['suspicious']:>6}\t{r['confirmed ZOs']:>6}\n"
            )

    print(f"\n  Summary -> {summary_path}")
    print(f"  Done. {len(all_results)}/{n_det} detectors processed.\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Euclid NISP 5-panel pipeline: '
            'raw | cont-sub | hot-removed | hot-noise | detections'
        )
    )

    parser.add_argument('fits_file', help='Path to the input FITS file')
    parser.add_argument(
        '--output-dir', '-o', help='Output directory (default: output/ next to the input file)',
        default=None
    )
    parser.add_argument(
        '--det-index', help='Zero-based detector index to process (default: all detectors)',
        type=int, default=None
    )
    parser.add_argument(
        '--kernel-rows', help='Continuum filter rows (default: 1)',
        type=int, default=1
    )
    parser.add_argument(
        '--kernel-cols', help='Continuum filter columns (default: 41)',
        type=int, default=41
    )
    parser.add_argument(
        '--inpaint-box', help='Box size for local background estimation during inpainting',
        type=int, default=64
    )
    parser.add_argument(
        '--hot-pixel-bits', help='Bitmask for selecting hot pixels from DQ array (default: 0xFFFFFFFF)',
        type=lambda x: int(x, 0), default=0xFFFFFFFF
    )
    parser.add_argument(
        '--intensity-threshold', help='Normalised detection threshold (default: 0.05)',
        type=float, default=0.05
    )
    parser.add_argument(
        '--detection-sigma', help='Sigma above background for detection (default: 3.0)',
        type=float, default=3.0
    )
    parser.add_argument(
        '--min-area', help='Minimum source area in pixels (default: 15)',
        type=int, default=15
    )
    parser.add_argument(
        '--fill-ratio-threshold', help='Fill ratio below which objects are marked suspicious (default: 0.20)',
        type=float, default=0.20
    )
    parser.add_argument(
        '--overlap-threshold',
        type=float, default=0.1
    )
    parser.add_argument(
        '--horizontal-overlap-threshold',
        type=float, default=0.3
    )
    parser.add_argument(
        '--proximity-threshold',
        type=int, default=30
    )
    parser.add_argument(
        '--aspect-ratio-tolerance',
        type=float, default=2.0
    )
    parser.add_argument(
        '--image-format', help='Output image format (default: jpeg)',
        type=str, default='jpeg', choices=['jpeg', 'jpg', 'png']
    )
    parser.add_argument(
        '--jpeg-quality', help='JPEG quality 1-95 (default: 85)',
        type=int, default=85
    )
    parser.add_argument(
        '--no-5panel', help='Skip saving the combined 5-panel image',
        action='store_true'
    )
    parser.add_argument(
        '--no-panel5-standalone', help='Skip saving the high-DPI panel 5 standalone image',
        action='store_true'
    )
    parser.add_argument(
        '--no-panel5-fits', help='Skip saving the panel 5 FITS file',
        action='store_true'
    )

    args = parser.parse_args()

    if not os.path.exists(args.fits_file):
        print(f"ERROR: file not found: {args.fits_file}")
        return
    
    zo_table = pd.read_csv(ESC_ALL_PATH)
    zo_table = zo_table[zo_table['Magnitude'] < ZO_MIN_BRIGHTNESS_MAG]

    process_fits_with_full_pipeline(
        fits_path          = args.fits_file,
        zo_table           = zo_table,
        output_dir         = args.output_dir,
        kernel_size        = (args.kernel_rows, args.kernel_cols),
        inpaint_box        = args.inpaint_box,
        hot_pixel_bits     = args.hot_pixel_bits,
        specific_det_index = args.det_index,
        image_format       = args.image_format,
        jpeg_quality       = args.jpeg_quality,
        detection_params   = {
            'intensity_threshold': args.intensity_threshold,
            'detection_sigma'    : args.detection_sigma,
            'min_area'           : args.min_area,
            'max_area'           : 100000,
            'max_objects'        : 1000,
        },
        merge_params       = {
            'overlap_threshold'           : args.overlap_threshold,
            'horizontal_overlap_threshold': args.horizontal_overlap_threshold,
            'proximity_threshold'         : args.proximity_threshold,
            'aspect_ratio_tolerance'      : args.aspect_ratio_tolerance,
        },
        filter_params      = {
            'min_aspect_ratio'    : 3.0,
            'min_width'           : 50,
            'fill_ratio_threshold': args.fill_ratio_threshold,
        },
        save_5panel              = not args.no_5panel,
        save_panel5_standalone   = not args.no_panel5_standalone,
        save_panel5_fits_flag    = not args.no_panel5_fits,
    )


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python full_pipeline_5panel.py <fits_file> [options]")
        print("       python full_pipeline_5panel.py --help")
    else:
        main()
