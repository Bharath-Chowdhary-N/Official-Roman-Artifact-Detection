# Euclid Artifact Detection Pipeline

This script processes Euclid NISP SCIFRM FITS files and produces a set of
diagnostic images and data products for each detector.

---

## What it does

For every detector (DETxx.SCI extension) in the input file the pipeline runs
through five stages, each of which becomes one panel in the output image.

**Panel 1 - Raw science image.** The raw pixel data displayed with a ZScale
stretch so faint structure is visible without saturation.

**Panel 2 - Continuum-subtracted image.** A running median along the dispersion
direction removes the spectral continuum, leaving point sources isolated. The
grism rotation angle is read from the FITS header so the filter is applied at
the correct orientation.

**Panel 3 - Hot pixels flagged.** Pixels marked bad in the DETxx.DQ quality
array are set to NaN and appear black. This shows you exactly where the
instrument artefacts are.

**Panel 4 - Hot pixels replaced.** Each bad pixel is replaced with a draw from
a Gaussian whose mean and width are estimated from the surrounding good pixels.
This makes the image usable for source detection without the spikes interfering.

**Panel 5 - Object detections.** Confident detections are outlined in green and
suspicious ones (low fill ratio, possibly residual artefacts) in yellow.
Detection uses a sigma-clipping threshold relative to the local background
noise, so it adapts to detector-to-detector variations automatically.

---

## Output files

Everything goes into the `output/` folder next to the input file by default,
or into whatever directory you specify with `--output-dir`. For each detector
you get:

- `*_5panel.jpg` (or `.png`) - the combined five-panel diagnostic image at 150 DPI
- `*_panel5.jpg` (or `.png`) - panel 5 alone at 300 DPI, good for publication or inspection
- `*_panel5.fits` - the inpainted continuum-subtracted image as a FITS file, with
  detection bounding boxes stored in a binary table extension called DETECTIONS
- `*_summary.txt` - a plain-text table summarising hot pixel counts and detection
  counts for every detector in the run

Any of the three image/FITS outputs can be turned off individually if you do not
need them (see the flags below).

---

## Requirements

### Main pipeline (`main.py`)

```
pip install numpy matplotlib astropy scipy scikit-image
```

### Upload pipeline (`upload_pipeline.py`)

```
pip install numpy matplotlib pandas astropy panoptes-client torch segment-anything
```

Also requires `MaskEncoder.py` to be present in the same directory.

On first run the SAM model checkpoint is downloaded automatically into `checkpoints/`:
- `vit_b` — 375 MB
- `vit_l` — 1.25 GB
- `vit_h` — 2.56 GB

### Caesar upload (`upload_detections_to_caesar.py`)

```
pip install pandas requests panoptes-client google-api-python-client google-auth-oauthlib
```

The Google Drive packages are only needed if using `--gdrive-credentials` for automated upload.

Python 3.9 or newer is recommended.

---

## Basic usage

Process all detectors in a file and write output to `output/`:

```bash
python main.py fits_file.fits
```

Process only the third detector (zero-based index):

```bash
python main.py fits_file.fits --det-index 2
```

Write output to a specific folder:

```bash
python main.py fits_file.fits --output-dir /data/results
```

Save as PNG instead of JPEG:

```bash
python main.py fits_file.fits --image-format png
```

---

## Turning off individual outputs

If you only want the FITS output and do not need the images:

```bash
python main.py fits_file.fits --no-5panel --no-panel5-standalone
```

If you only want the images and not the FITS file:

```bash
python main.py fits_file.fits --no-panel5-fits
```

Skip everything except the summary text file (useful for a quick statistics run):

```bash
python main.py fits_file.fits --no-5panel --no-panel5-standalone --no-panel5-fits
```

---

## Detection tuning

The default settings work well for typical Euclid NISP data but you may want to
adjust them for very crowded fields or very faint targets.

`--detection-sigma` sets how many standard deviations above the background a
pixel needs to be before it is considered part of a source. The default is 3.0.
Lower values find fainter sources but also pick up more noise.

`--min-area` controls the smallest source in pixels. The default is 15. Raise
this if you are getting lots of spurious single-pixel detections.

`--fill-ratio-threshold` separates confident detections from suspicious ones. An
object whose filled pixels cover less than this fraction of its bounding box gets
a yellow box instead of green. The default is 0.20.

`--proximity-threshold` is the maximum distance in pixels between two separate
detections before they are merged into one. The default is 30.

---

## All options

```
positional arguments:
  fits_file                      Path to the input FITS file

optional arguments:
  -o, --output-dir               Output directory (default: output/ next to input)
  --det-index                    Zero-based detector index (default: all)
  --kernel-rows                  Continuum filter height (default: 1)
  --kernel-cols                  Continuum filter width (default: 41)
  --inpaint-box                  Box size for background estimation (default: 64)
  --hot-pixel-bits               DQ bitmask for bad pixels (default: 0xFFFFFFFF)
  --intensity-threshold          Normalised detection threshold (default: 0.05)
  --detection-sigma              Detection threshold in sigma (default: 3.0)
  --min-area                     Minimum source area in pixels (default: 15)
  --fill-ratio-threshold         Suspicious object cutoff (default: 0.20)
  --overlap-threshold            Merge overlap threshold (default: 0.1)
  --horizontal-overlap-threshold Horizontal overlap for merging (default: 0.3)
  --proximity-threshold          Merge proximity in pixels (default: 30)
  --aspect-ratio-tolerance       Max aspect ratio difference for merging (default: 2.0)
  --image-format                 jpeg or png (default: jpeg)
  --jpeg-quality                 JPEG quality 1-95 (default: 85)
  --no-5panel                    Skip the combined 5-panel image
  --no-panel5-standalone         Skip the high-DPI panel 5 image
  --no-panel5-fits               Skip the panel 5 FITS output
```

---

## The FITS output in detail

The panel 5 FITS file has two extensions. The primary extension contains the
inpainted continuum-subtracted image as 32-bit floats, with a few header keywords
summarising the grism angle, hot pixel count, and detection counts.

The second extension, named DETECTIONS, is a binary table with one row per
detected object. The columns are X_MIN, Y_MIN, X_MAX, Y_MAX (bounding box corners
in pixels), CENT_X and CENT_Y (flux-weighted centroid), AREA (number of source
pixels), FILL (fill ratio), ASPECT (aspect ratio), and TYPE (either "confident"
or "suspicious").

---

## Uploading to Caesar / Zooniverse

Two upload scripts push pipeline outputs to the Zooniverse citizen-science platform.

**Run order:**

1. `main.py` — detects objects and writes `*_panel5.fits` and `*_detections.csv`
2. `upload_pipeline.py` — runs SAM on the detections and uploads subjects to Zooniverse; also creates `subject_set_registry.json`
3. `upload_detections_to_caesar.py` — depends on `subject_set_registry.json` produced in step 2

**Runtime files needed before uploading:**

| File | Created by | Purpose |
|------|-----------|---------|
| `*_panel5.fits` | `main.py` | Input image for SAM and Caesar |
| `*_detections.csv` | `main.py` | Detection bounding boxes |
| `subject_set_registry.json` | `upload_pipeline.py` (auto) | Caches Zooniverse subject set IDs |
| `gdrive_oauth_cache.json` | Google Cloud Console (manual) | OAuth credentials for Google Drive |
| `gdrive_auth_token.json` | First OAuth login (auto after) | Cached Google Drive access token |

### upload_detections_to_caesar.py

Uploads a panel-5 FITS file and its detection CSV to a Zooniverse subject set, with
Google Drive backing for the subject images.

```bash
python upload_detections_to_caesar.py \
  --fits Images/output/EUC_SIR_W-SCIFRM_BKGSUB_2681_11832_3_RGS180_0_2024-10-31T17:44:11.173589Z_det0_DET11_SCI_panel5.fits \
  --detections Images/output/EUC_SIR_W-SCIFRM_BKGSUB_2681_11832_3_RGS180_0_2024-10-31T17:44:11.173589Z_det0_detections.csv \
  --registry subject_set_registry.json \
  --workflow-id 31550 \
  --username YOUR_USERNAME \
  --password 'YOUR_PASSWORD' \
  --file-id 1nhrZSrPEQCErS0S2W6qRTCP22LH76ZGH \
  --gdrive-token gdrive_auth_token.json \
  --gdrive-credentials gdrive_oauth_cache.json
```

| Flag | Description |
|------|-------------|
| `--fits` | Panel-5 FITS file produced by `main.py` |
| `--detections` | Detection CSV produced by `main.py` |
| `--registry` | JSON file tracking uploaded subject sets (created on first run) |
| `--workflow-id` | Zooniverse workflow ID to attach subjects to |
| `--username` / `--password` | Zooniverse credentials |
| `--file-id` | Google Drive file ID for the subject image |
| `--gdrive-token` | Cached OAuth token for Google Drive |
| `--gdrive-credentials` | OAuth credentials file for Google Drive |

---

### upload_pipeline.py

Runs the Segment Anything Model (SAM) over the detections and uploads the resulting
segmentation masks to a Zooniverse project.

```bash
python upload_pipeline.py \
  --fits Images/output/EUC_SIR_W-SCIFRM_BKGSUB_2681_11832_3_RGS180_0_2024-10-31T17:44:11.173589Z_det0_DET11_SCI_panel5.fits \
  --detections Images/output/EUC_SIR_W-SCIFRM_BKGSUB_2681_11832_3_RGS180_0_2024-10-31T17:44:11.173589Z_det0_detections.csv \
  --username YOUR_USERNAME \
  --password 'YOUR_PASSWORD' \
  --model-type vit_h \
  --project-id 30908
```

| Flag | Description |
|------|-------------|
| `--fits` | Panel-5 FITS file produced by `main.py` |
| `--detections` | Detection CSV produced by `main.py` |
| `--username` / `--password` | Zooniverse credentials |
| `--model-type` | SAM model variant: `vit_h`, `vit_l`, or `vit_b` (default: `vit_h`) |
| `--project-id` | Zooniverse project ID to upload subjects to |

---

### Setting up gdrive_auth_token.json

`gdrive_auth_token.json` is the cached OAuth token that lets the script write to
Google Drive without prompting for a browser login every time. Generate it once:

1. Make sure you have a `gdrive_oauth_cache.json` credentials file downloaded from
   the [Google Cloud Console](https://console.cloud.google.com/) for a project with
   the Drive API enabled (OAuth 2.0 Desktop App credentials).

2. Run the one-time authentication helper:

   ```bash
   python -c "
   from google_auth_oauthlib.flow import InstalledAppFlow
   import json, pickle

   SCOPES = ['https://www.googleapis.com/auth/drive']
   flow = InstalledAppFlow.from_client_secrets_file('gdrive_oauth_cache.json', SCOPES)
   creds = flow.run_local_server(port=0)
   with open('gdrive_auth_token.json', 'w') as f:
       f.write(creds.to_json())
   print('Token saved to gdrive_auth_token.json')
   "
   ```

3. A browser window will open asking you to authorise access. After approval,
   `gdrive_auth_token.json` is written to the current directory. The token is
   refreshed automatically on subsequent runs as long as the file is present.

> **Note:** Keep both JSON files out of version control — add them to `.gitignore`.

---


