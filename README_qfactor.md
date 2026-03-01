# qfactor.py

Standalone module for computing Q-factors from OME-TIFF z-map images. Encapsulates the full processing pipeline with optimal parameters from the study.

---

## Requirements

```bash
pip install numpy pandas pillow scipy tqdm
```

---

## Public API

### `compute_qfactor(filepath, segment_width_um=1.0, params=None) → DataFrame`

Processes a single z-map file. Returns one row per slice.

```python
from qfactor import compute_qfactor

df = compute_qfactor("scan_001.tif", segment_width_um=1.0)
```

### `process_dataset(zmap_files, segment_width_um=1.0, params=None) → DataFrame`

Batch-processes a list of files. Returns all slices from all files concatenated.

```python
from qfactor import process_dataset

df = process_dataset(all_zmap_files, segment_width_um=1.0)
```

`zmap_files` is a list of `(directory_path, filename)` tuples — the same format produced by the notebook's data loading cell.

---

## Output

One row per slice. Shape: `(n_total_slices, 13)`.

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | Source file name |
| `slice_id` | int | Slice index within the image |
| `position_um` | float | Physical start position of the slice (µm) |
| `q_factor` | float | Q-factor (`NaN` if calculation failed) |
| `A` | float | Ablated area (nm·µm) |
| `R` | float | Redeposited area (nm·µm) |
| `baseline` | float | Estimated baseline height (nm) |
| `sigma_shoulder` | float | Shoulder roughness (nm) |
| `orientation` | str | `'horizontal'` or `'vertical'` |
| `inverted` | bool | Whether z-axis was flipped |
| `trench_start` | float | Trench region start index (pixels) |
| `trench_end` | float | Trench region end index (pixels) |
| `pixel_size_um` | float | Pixel size of source image (µm/pixel) |

No filtering is applied — all slices are returned. Failed slices have `NaN` in `q_factor`.

---

## Parameters

### `segment_width_um`

Controls slice width in micrometers. The only parameter you need to tune for most use cases.

```python
df = compute_qfactor("scan.tif", segment_width_um=0.5)  # finer slices
df = compute_qfactor("scan.tif", segment_width_um=2.0)  # coarser slices
```

### `QFactorParams`

Controls the Q-factor calculation itself. Defaults are the optimal values from the study — only change if experimenting.

```python
from qfactor import QFactorParams

params = QFactorParams(
    alpha           = 0.15,      # shoulder width fraction for baseline estimation
    baseline_method = "median",  # "median" or "mean"
    gaussian_sigma  = 2.0,       # smoothing applied to profile (0 = disabled)
    window_um       = 20.0,      # trench detection window radius in µm (critical)
)

df = compute_qfactor("scan.tif", params=params)
```

---

## Pipeline

Every image goes through these steps in order:

```
1. extract_pixel_size()         reads µm/pixel from OME-XML metadata
2. detect_trench_orientation()  determines horizontal or vertical trench
3. check_inversion_needed()     flips z-axis if trench appears as a peak
4. segment_zmap()               slices image into strips of segment_width_um
5. extract_profile()            averages each strip into a 1D profile
6. gaussian_filter1d()          smooths profile (sigma = gaussian_sigma)
7. calculate_baseline()         estimates baseline from shoulder regions
8. detect_trench_region()       finds trench as window around profile minimum
9. calculate_areas()            computes A and R within trench region
10. Q = 1 - R / A
```

---

## ML Usage

```python
from qfactor import process_dataset

# Compute Q-factors for all images
df_q = process_dataset(all_zmap_files, segment_width_um=1.0)

# Join with process metadata on filename
df = df_q.merge(df_metadata, on="filename")

# Drop failed slices
df = df[df["q_factor"].notna()]

# Build ML dataset
X = df[["A", "R", "baseline", "sigma_shoulder", "laser_power", ...]]
y = df["q_factor"]
```
