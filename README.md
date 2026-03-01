# Q-Factor Analysis — Laser-Etched Trenches

Quality analysis of laser-etched trench surfaces using Q-factor as the quality metric.
Dataset: 423 z-map images, Silicon Wafer, varying scan speed, number of passes and range.

---

## What is Q-Factor?

```
Q = 1 - R / A
```

| Symbol | Meaning |
|--------|---------|
| A | Ablated area — material removed below baseline (nm·µm) |
| R | Redeposited area — material above baseline inside trench (nm·µm) |
| Q = 1 | Perfect trench |
| Q = 0 | Fully redeposited |
| Q < 0 | Failed trench |

---

## Setup

```bash
pip install numpy pandas matplotlib pillow scipy tqdm gdown openpyxl scikit-learn
```

---

## Project Structure

```
├── analysis.ipynb          ← final analysis notebook (start here)
├── src/
│   ├── qfactor.py          ← Q-factor computation module
│   └── extract_qfactors.py ← batch CSV extraction script
├── notebooks/
│   ├── q-factor-study.ipynb ← parameter optimization study
│   ├── eda.ipynb            ← exploratory data analysis
│   ├── regression.ipynb     ← curve fitting & regression
│   └── stats.ipynb          ← statistical analysis
├── README.md
└── README_qfactor.md       ← qfactor.py API documentation
```

---

## Workflow

**Step 1 — Extract Q-factors from z-map images:**
```bash
python src/extract_qfactors.py              # default 1.0 µm slices
python src/extract_qfactors.py --slice 0.5  # custom slice width
```
Outputs `qfactors_1.0um.csv` (all slices with Q-factor, A, R and metadata).

**Step 2 — Run the analysis:**

Open `analysis.ipynb` and run top-to-bottom. Exports figures and a summary CSV to `results/`.

---

## Key Findings

- **A and R are predictable** from process parameters (CV R² ≈ 0.49 / 0.43)
- **Q is not** — A and R co-vary, so the ratio R/A cancels most of the signal
- **scan_speed** is the key lever for Q: faster scanning → less redeposition
- **n_pass and range_um** drive total material removal (volume)
- Q is spatially uniform along the trench; edge effects are minimal

---

## Optimal Q-Factor Parameters

From `notebooks/q-factor-study.ipynb`:

| Parameter | Value | Impact |
|-----------|-------|--------|
| Baseline method | Median | Minimal |
| Shoulder width α | 0.15 | Minimal |
| Gaussian σ | 2.0 | Small |
| Window size | 20 µm | **Critical** |
