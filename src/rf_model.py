"""
rf_model.py

Random Forest prediction of laser trench quality metrics (A, R, Q)
from process parameters.

Methodology (following Zhang et al. 2022):
  - Keep only parameter combos with >1 image (replicated experiments)
  - Sample 3 random slices per image for A, R, Q values
  - Standardized inputs (fit on training set only)
  - 100 random 80/20 train/test splits
  - Grid search for hyperparameters per split
  - Report mean +/- std of R2 and MAE

Usage
-----
    python src/rf_model.py                     # full run (100 splits)
    python src/rf_model.py --n-splits 10       # quick test
    python src/rf_model.py --n-slices 5        # 5 slices per image
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
FEATURES = ["scan_speed", "n_pass"]
POWER_FILTER = 1.85  # Keep only this power level (most replicated data)
TEST_SIZE = 0.2
N_RANDOM_SPLITS = 100
N_SLICES_PER_IMAGE = 3

RENAME_MAP = {
    "Power(W)": "power",
    "Scanning speed(mm/s)": "scan_speed",
    "Pass": "n_pass",
}

RF_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path, n_slices: int = 3, seed: int = 42):
    """
    Load dataset:
      1. Keep only combos with >1 image (replicated experiments)
      2. Sample n_slices random slices per image
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RENAME_MAP)

    # Filter valid Q and single power level
    valid = df["q_factor"].notna() & df["q_factor"].between(0, 1)
    df = df.loc[valid].copy()
    df = df[df["power"] == POWER_FILTER].copy()

    # Keep only combos with >1 image
    combo_counts = df.groupby(["scan_speed", "n_pass"])["filename"].nunique()
    good_combos = combo_counts[combo_counts > 1].index
    df = df.set_index(["scan_speed", "n_pass"])
    df = df.loc[df.index.isin(good_combos)].reset_index()

    # Sample n_slices random slices per image
    rng = np.random.RandomState(seed)
    sampled = df.groupby("filename").apply(
        lambda g: g.sample(n=min(n_slices, len(g)), random_state=rng),
        include_groups=False,
    ).reset_index(level=0, drop=False)

    n_images = df["filename"].nunique()
    n_combos = len(good_combos)

    X = sampled[FEATURES].values
    targets = {
        "A": sampled["A"].values,
        "R": sampled["R"].values,
        "Q": sampled["q_factor"].values,
    }
    groups = sampled["filename"].values

    return X, targets, groups, n_combos, n_images


# ---------------------------------------------------------------------------
# Core: train & evaluate over N random splits
# ---------------------------------------------------------------------------

def evaluate(X, y, groups, n_splits):
    """
    N random 80/20 splits (grouped by image to prevent leakage).
    Each split: standardize, grid search RF, evaluate train+test.
    """
    r2_train, r2_test = [], []
    mae_train, mae_test = [], []
    best_model = None
    best_r2 = -np.inf
    best_data = None

    # Get unique images for splitting
    unique_images = np.unique(groups)

    for i in tqdm(range(n_splits), desc="  Splits", leave=False):
        # Split by image (not by row) to prevent leakage
        img_train, img_test = train_test_split(
            unique_images, test_size=TEST_SIZE, random_state=i)

        train_mask = np.isin(groups, img_train)
        test_mask = np.isin(groups, img_test)

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        # Standardize
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Grid search
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
    ax.set_ylabel("R2 (%)")
    ax.set_title("Random Forest - R2 (100 random splits)")
    ax.legend()
    ax.set_ylim(0, 110)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overfitting(results_dict, out_path):
    targets = list(results_dict.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4),
                             constrained_layout=True)
    for ax, t in zip(axes, targets):
        overfit = (results_dict[t]["r2_train"] - results_dict[t]["r2_test"]) * 100
        ax.hist(overfit, bins=15, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Train R2 - Test R2 (%)")
        ax.set_ylabel("Count")
        ax.set_title("Overfitting - %s" % t)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(y_true, y_pred, target_name, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
    ax.set_xlabel("Experimental %s" % target_name)
    ax.set_ylabel("Predicted %s" % target_name)
    r2 = r2_score(y_true, y_pred)
    ax.set_title("RF - %s (best split R2=%.3f)" % (target_name, r2))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RF prediction of A, R, Q")
    parser.add_argument("--n-splits", type=int, default=N_RANDOM_SPLITS,
                        help="Number of random 80/20 splits (default: 100)")
    parser.add_argument("--n-slices", type=int, default=N_SLICES_PER_IMAGE,
                        help="Random slices sampled per image (default: 3)")
    parser.add_argument("--out", type=str, default="results/rf",
                        help="Output directory (default: results/rf)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, targets, groups, n_combos, n_images = load_data(
        CSV_PATH, n_slices=args.n_slices)

    print("Random Forest - Laser Trench Quality Prediction")
    print("=" * 50)
    print("Combos (>1 img): %d" % n_combos)
    print("Images          : %d" % n_images)
    print("Slices/image    : %d" % args.n_slices)
    print("Total rows      : %d" % len(X))
    print("Features        : %s" % ", ".join(FEATURES))
    print("Normalization   : StandardScaler (fit on train)")
    print("Split strategy  : %d random 80/20 (grouped by image)" % args.n_splits)
    print("Hyperparam      : GridSearchCV (3-fold on train)")
    print()

    all_results = {}
    summary = []

    for name, y in targets.items():
        print("[%s] Running %d splits..." % (name, args.n_splits))
        res = evaluate(X, y, groups, args.n_splits)
        all_results[name] = res

        r2_tr, r2_te = res["r2_train"], res["r2_test"]
        mae_tr, mae_te = res["mae_train"], res["mae_test"]

        print("  R2 train=%.1f+/-%.1f%%  test=%.1f+/-%.1f%%  "
              "MAE train=%.3f+/-%.3f  test=%.3f+/-%.3f" % (
                  r2_tr.mean()*100, r2_tr.std()*100,
                  r2_te.mean()*100, r2_te.std()*100,
                  mae_tr.mean(), mae_tr.std(),
                  mae_te.mean(), mae_te.std()))

        summary.append({
            "target": name,
            "r2_train": "%.1f+/-%.1f" % (r2_tr.mean()*100, r2_tr.std()*100),
            "r2_test": "%.1f+/-%.1f" % (r2_te.mean()*100, r2_te.std()*100),
            "mae_train": "%.3f+/-%.3f" % (mae_tr.mean(), mae_tr.std()),
            "mae_test": "%.3f+/-%.3f" % (mae_te.mean(), mae_te.std()),
            "overfit": "%.1f%%" % ((r2_tr - r2_te).mean()*100),
        })

        y_te, pred_te, _ = res["best_data"]
        plot_pred_vs_actual(y_te, pred_te, name,
                            out_dir / ("pred_vs_actual_%s.png" % name))

    plot_r2_bar(all_results, out_dir / "r2_comparison.png")
    plot_overfitting(all_results, out_dir / "overfitting.png")
    pd.DataFrame(summary).to_csv(out_dir / "results.csv", index=False)

    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nResults saved to %s/" % out_dir)


if __name__ == "__main__":
    main()
