"""
rf_model.py

Random Forest prediction of laser trench quality metrics (A, R, Q)
from process parameters.

Methodology (following Zhang et al. 2022):
  - One row per image (median A, R, Q)
  - Standardized inputs (fit on training set only)
  - 100 random 80/20 train/test splits
  - Grid search for hyperparameters per split
  - Report mean ± std of R² and MAE

Usage
-----
    python src/rf_model.py                     # full run (100 splits)
    python src/rf_model.py --n-splits 10       # quick test
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH = Path("qfactors_1.0um.csv")
FEATURES = ["scan_speed", "n_pass", "range_um", "power"]
TEST_SIZE = 0.2
N_RANDOM_SPLITS = 100

RENAME_MAP = {
    "Power(W)": "power",
    "Scanning speed(mm/s)": "scan_speed",
    "Pass": "n_pass",
    "Range(micron)": "range_um",
}

RF_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path):
    """Load dataset, aggregate to one row per image (median A, R, Q)."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RENAME_MAP)

    valid = df["q_factor"].notna() & df["q_factor"].between(0, 1)
    df = df.loc[valid].copy()

    agg = df.groupby("filename").agg(
        A=("A", "median"),
        R=("R", "median"),
        Q=("q_factor", "median"),
        scan_speed=("scan_speed", "first"),
        n_pass=("n_pass", "first"),
        range_um=("range_um", "first"),
        power=("power", "first"),
    ).reset_index()

    X = agg[FEATURES].values
    targets = {"A": agg["A"].values, "R": agg["R"].values, "Q": agg["Q"].values}
    return X, targets


# ---------------------------------------------------------------------------
# Core: train & evaluate over N random splits
# ---------------------------------------------------------------------------

def evaluate(X, y, n_splits):
    """
    100 random 80/20 splits. Each split:
      1. Standardize (fit on train only)
      2. Grid search RF hyperparams (3-fold CV on train)
      3. Evaluate on train and test
    """
    r2_train, r2_test = [], []
    mae_train, mae_test = [], []
    best_model = None
    best_r2 = -np.inf
    best_data = None

    for i in tqdm(range(n_splits), desc="  Splits", leave=False):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=i)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            RF_PARAM_GRID, cv=3, scoring="r2", n_jobs=-1,
        )
        grid.fit(X_tr_s, y_tr)
        est = grid.best_estimator_

        pred_tr = est.predict(X_tr_s)
        pred_te = est.predict(X_te_s)

        r2_tr = r2_score(y_tr, pred_tr)
        r2_te = r2_score(y_te, pred_te)
        r2_train.append(r2_tr)
        r2_test.append(r2_te)
        mae_train.append(mean_absolute_error(y_tr, pred_tr))
        mae_test.append(mean_absolute_error(y_te, pred_te))

        if r2_te > best_r2:
            best_r2 = r2_te
            best_model = est
            best_data = (y_te, pred_te, scaler)

    return {
        "r2_train": np.array(r2_train),
        "r2_test": np.array(r2_test),
        "mae_train": np.array(mae_train),
        "mae_test": np.array(mae_test),
        "best_model": best_model,
        "best_data": best_data,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_r2_bar(results_dict, out_path):
    """R² bar chart (train vs test) per target."""
    targets = list(results_dict.keys())
    train_m = [results_dict[t]["r2_train"].mean() * 100 for t in targets]
    test_m = [results_dict[t]["r2_test"].mean() * 100 for t in targets]
    train_s = [results_dict[t]["r2_train"].std() * 100 for t in targets]
    test_s = [results_dict[t]["r2_test"].std() * 100 for t in targets]

    x = np.arange(len(targets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.bar(x - w/2, train_m, w, yerr=train_s, label="Training", capsize=4)
    ax.bar(x + w/2, test_m, w, yerr=test_s, label="Testing", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("R² (%)")
    ax.set_title("Random Forest — R² (100 random splits)")
    ax.legend()
    ax.set_ylim(0, 110)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overfitting(results_dict, out_path):
    """Overfitting histogram per target."""
    targets = list(results_dict.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4),
                             constrained_layout=True)
    for ax, t in zip(axes, targets):
        overfit = (results_dict[t]["r2_train"] - results_dict[t]["r2_test"]) * 100
        ax.hist(overfit, bins=15, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Train R² − Test R² (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"Overfitting — {t}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(y_true, y_pred, target_name, out_path):
    """Best-split scatter plot."""
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
    ax.set_xlabel(f"Experimental {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"RF — {target_name} (best split R²={r2:.3f})")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RF prediction of A, R, Q")
    parser.add_argument("--n-splits", type=int, default=N_RANDOM_SPLITS,
                        help="Number of random 80/20 splits (default: 100)")
    parser.add_argument("--out", type=str, default="results/rf",
                        help="Output directory (default: results/rf)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, targets = load_data(CSV_PATH)

    print("Random Forest — Laser Trench Quality Prediction")
    print("=" * 50)
    print(f"Samples       : {len(X)} (one per image)")
    print(f"Features      : {', '.join(FEATURES)}")
    print(f"Normalization : StandardScaler (fit on train)")
    print(f"Strategy      : {args.n_splits} random 80/20 splits")
    print(f"Hyperparam    : GridSearchCV (3-fold on train)")
    print()

    all_results = {}
    summary = []

    for name, y in targets.items():
        print(f"[{name}] Running {args.n_splits} splits...", end=" ", flush=True)
        res = evaluate(X, y, args.n_splits)
        all_results[name] = res

        r2_tr, r2_te = res["r2_train"], res["r2_test"]
        mae_tr, mae_te = res["mae_train"], res["mae_test"]

        print(f"R² train={r2_tr.mean()*100:.1f}±{r2_tr.std()*100:.1f}%  "
              f"test={r2_te.mean()*100:.1f}±{r2_te.std()*100:.1f}%  "
              f"MAE train={mae_tr.mean():.3f}±{mae_tr.std():.3f}  "
              f"test={mae_te.mean():.3f}±{mae_te.std():.3f}")

        summary.append({
            "target": name,
            "r2_train": f"{r2_tr.mean()*100:.1f}±{r2_tr.std()*100:.1f}",
            "r2_test": f"{r2_te.mean()*100:.1f}±{r2_te.std()*100:.1f}",
            "mae_train": f"{mae_tr.mean():.3f}±{mae_tr.std():.3f}",
            "mae_test": f"{mae_te.mean():.3f}±{mae_te.std():.3f}",
            "overfit": f"{(r2_tr - r2_te).mean()*100:.1f}%",
        })

        # Pred vs actual for best split
        y_te, pred_te, _ = res["best_data"]
        plot_pred_vs_actual(y_te, pred_te, name,
                            out_dir / f"pred_vs_actual_{name}.png")

    # Plots
    plot_r2_bar(all_results, out_dir / "r2_comparison.png")
    plot_overfitting(all_results, out_dir / "overfitting.png")

    # Save CSV
    pd.DataFrame(summary).to_csv(out_dir / "results.csv", index=False)

    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
