# Q-Factor Parameter Optimization Study

Parameter optimization study for Q-factor calculation on OME-TIFF z-map images of laser-etched trenches. Determines the best configuration for quantifying trench quality. Results feed into `qfactor.py`, a standalone module used for ML dataset preparation.

---

## What is Q-Factor?

```
Q = 1 - R / A
```

- **A** — ablated area (material removed below baseline)
- **R** — redeposited area (material above baseline inside trench)
- **Q = 1** → perfect trench, **Q = 0** → fully redeposited, **Q < 0** → failed trench

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn pillow scipy tqdm gdown openpyxl
```

Set your Google Drive sharing URLs in **Cell 1**:

```python
DRIVE_URLS = {
    "06082025": "https://drive.google.com/file/d/.../view?usp=sharing",
    "26082025": "https://drive.google.com/file/d/.../view?usp=sharing",
}
```

Data is downloaded once and skipped on subsequent runs.

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Imports, `Config` class, Drive URLs |
| 2 | Download from Google Drive, extract, load file list |
| 3 | Load sample z-map, detect orientation, visualize |
| 4 | Segment z-map into 1 µm strips |
| 5 | Extract 1D profiles from strips |
| 7 | Baseline detection and Q-factor calculation |
| 8 | Batch processing over all files |
| 9 | Quality filtering |
| 10 | Visual inspection by Q-factor range |
| 12 | Experiment: baseline method (Median vs Mean) |
| 13 | Experiment: shoulder width α (0.10, 0.15, 0.20) |
| 14 | Experiment: Gaussian smoothing σ (0, 1, 2, 3, 5) |
| 15 | Experiment: trench window size (15, 20, 25, 30 µm) |
| 16 | Final optimal configuration summary |
| 17 | Report figures |

---

## Optimal Parameters

| Parameter | Value | Impact |
|-----------|-------|--------|
| Baseline method | Median | Minimal |
| Shoulder width α | 0.15 | Minimal |
| Gaussian σ | 2.0 | Small |
| Window size | 20 µm | **Critical** |

---

## Output

The standalone `qfactor.py` module is the output of this study. See `README_qfactor.md`.
