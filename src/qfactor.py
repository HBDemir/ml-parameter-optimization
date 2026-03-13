"""
Q-Factor computation module for z-map trench analysis.

Optimal parameters (from parameter optimization study):
    - Baseline method : median, alpha = 0.15
    - Gaussian smoothing : sigma = 2.0
    - Trench window size : 20 µm  (critical parameter)
    - Segment width : 1 µm
    - Quality filter threshold : sigma_shoulder < 0.6

Public API
----------
compute_qfactor(filepath, params=None) -> float | None
    Returns the median Q-factor across all valid segments for one image.

process_dataset(zmap_files, params=None) -> pd.DataFrame
    Batch-processes a list of (path, filename) tuples and returns a DataFrame.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# GPU support: use CuPy when USE_GPU=1 is set in the environment
_USE_GPU = os.environ.get('USE_GPU', '0') == '1'
try:
    if not _USE_GPU:
        raise ImportError
    import cupy as _cp
    from cupyx.scipy.ndimage import gaussian_filter1d as _cp_gauss
    def gaussian_filter1d(arr, sigma):
        return _cp.asnumpy(_cp_gauss(_cp.asarray(arr), sigma))
    print('qfactor: GPU backend active (CuPy)')
except ImportError:
    from scipy.ndimage import gaussian_filter1d
    if _USE_GPU:
        print('qfactor: CuPy not found, falling back to CPU')


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class QFactorParams:
    """
    Q-factor calculation parameters (optimal values from the optimisation study).

    Note: segment_width_um is NOT here — it is a top-level argument of
    compute_qfactor() and process_dataset() because it controls how the
    image is sliced, not how Q is calculated.
    """
    alpha: float = 0.15            # shoulder width fraction for baseline estimation
    baseline_method: str = "median" # "median" or "mean"
    gaussian_sigma: float = 2.0    # Gaussian smoothing applied to profile (0 = off)
    window_um: float = 20.0        # trench detection window radius in µm (critical)


DEFAULT_PARAMS = QFactorParams()


# ---------------------------------------------------------------------------
# Metadata & loading
# ---------------------------------------------------------------------------

def extract_pixel_size(filepath) -> Optional[float]:
    """Extract physical pixel size (µm/pixel) from OME-XML metadata in TIFF."""
    img = Image.open(filepath)
    xml_string = {k: v for k, v in img.tag_v2.items()}.get(270)
    if not xml_string:
        return None
    root = ET.fromstring(xml_string.encode("utf-8"))
    for elem in root.iter():
        if elem.tag.endswith("Pixels"):
            val = elem.get("PhysicalSizeX")
            if val:
                return float(val)
    return None


def load_zmap(filepath) -> np.ndarray:
    """Load a TIFF z-map as a numpy array."""
    return np.array(Image.open(str(filepath)))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def detect_trench_orientation(zmap: np.ndarray) -> str:
    """
    Return 'horizontal' if the trench runs left-right, 'vertical' otherwise.
    The axis with higher variance in the mean profile is perpendicular to the trench.
    """
    var_y = np.var(zmap.mean(axis=1))
    var_x = np.var(zmap.mean(axis=0))
    return "horizontal" if var_y > var_x else "vertical"


def check_inversion_needed(zmap: np.ndarray, orientation: str) -> bool:
    """Return True if the z-map needs to be flipped (trench appears as a peak)."""
    if orientation == "horizontal":
        h = zmap.shape[0]
        e = int(h * 0.3)
        trench = zmap[e:-e, :]
        rest = np.concatenate([zmap[:e, :], zmap[-e:, :]])
    else:
        w = zmap.shape[1]
        e = int(w * 0.3)
        trench = zmap[:, e:-e]
        rest = np.concatenate([zmap[:, :e], zmap[:, -e:]], axis=1)
    return trench.mean() > rest.mean()


# ---------------------------------------------------------------------------
# Segmentation & profile extraction
# ---------------------------------------------------------------------------

def segment_zmap(
    zmap: np.ndarray,
    pixel_size_um: float,
    orientation: str,
    segment_width_um: float = 1.0,
) -> list[np.ndarray]:
    """Slice z-map into strips perpendicular to the trench direction."""
    w_px = max(1, int(round(segment_width_um / pixel_size_um)))
    if orientation == "horizontal":
        return [zmap[:, i:i + w_px] for i in range(0, zmap.shape[1], w_px)
                if zmap[:, i:i + w_px].shape[1] > 0]
    else:
        return [zmap[i:i + w_px, :] for i in range(0, zmap.shape[0], w_px)
                if zmap[i:i + w_px, :].shape[0] > 0]


def extract_profile(segment: np.ndarray, orientation: str) -> np.ndarray:
    """Average a segment across its narrow dimension to get a 1-D profile."""
    return segment.mean(axis=1) if orientation == "horizontal" else segment.mean(axis=0)


# ---------------------------------------------------------------------------
# Baseline & Q-factor calculation
# ---------------------------------------------------------------------------

def calculate_baseline(
    profile: np.ndarray,
    alpha: float = 0.15,
    method: str = "median",
) -> tuple[float, dict]:
    """
    Estimate baseline from shoulder regions.

    b(h; α) = ½ [f(S_L) + f(S_R)]
    S_L = profile[:floor(αN)],  S_R = profile[ceil((1-α)N):]
    f = median or mean
    """
    N = len(profile)
    l_idx = int(np.floor(alpha * N))
    r_idx = int(np.ceil((1 - alpha) * N))
    left, right = profile[:l_idx], profile[r_idx:]

    fn = np.median if method == "median" else np.mean
    b_left, b_right = fn(left), fn(right)
    baseline = (b_left + b_right) / 2

    stats = {
        "b_left": b_left,
        "b_right": b_right,
        "sigma_left": np.std(left),
        "sigma_right": np.std(right),
        "sigma_shoulder": (np.std(left) + np.std(right)) / 2,
    }
    return baseline, stats


def detect_trench_region(
    profile: np.ndarray,
    baseline: float,
    pixel_size_um: float,
    window_um: float = 20.0,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Fixed window around the deepest point.
    Returns a boolean mask and (start, end) indices.
    """
    N = len(profile)
    half = int(window_um / pixel_size_um)
    min_idx = int(np.argmin(profile))
    start = max(0, min_idx - half)
    end = min(N, min_idx + half + 1)
    mask = np.zeros(N, dtype=bool)
    mask[start:end] = True
    return mask, (start, end)


def calculate_areas(
    profile: np.ndarray,
    baseline: float,
    delta_x: float,
    trench_mask: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """
    A = Δx · Σ [b − h(x)]₊   (ablated)
    R = Δx · Σ [h(x) − b]₊   (redeposited)
    Both summed only within the trench region.
    """
    if trench_mask is None:
        trench_mask = np.ones(len(profile), dtype=bool)
    region = profile[trench_mask]
    A = delta_x * np.sum(np.maximum(baseline - region, 0))
    R = delta_x * np.sum(np.maximum(region - baseline, 0))
    return A, R


def _qfactor_from_profile(
    profile: np.ndarray,
    pixel_size_um: float,
    params: QFactorParams,
) -> dict:
    """
    Calculate Q-factor for a single 1-D profile.
    Returns a result dict (or empty dict if calculation fails).
    """
    baseline, shoulder_stats = calculate_baseline(
        profile, params.alpha, params.baseline_method
    )
    trench_mask, (t_start, t_end) = detect_trench_region(
        profile, baseline, pixel_size_um, params.window_um
    )
    A, R = calculate_areas(profile, baseline, pixel_size_um, trench_mask)

    q = (1 - R / A) if A > 0 else np.nan

    return {
        "q_factor": q,
        "baseline": baseline,
        "sigma_shoulder": shoulder_stats["sigma_shoulder"],
        "A": A,
        "R": R,
        "trench_start": t_start,
        "trench_end": t_end,
    }


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_single_zmap(
    filepath,
    pixel_size_um: float,
    segment_width_um: float,
    params: QFactorParams = DEFAULT_PARAMS,
) -> list[dict]:
    """
    Full pipeline for one z-map file.
    Returns a list of per-segment result dicts.
    """
    zmap = load_zmap(filepath)
    orientation = detect_trench_orientation(zmap)
    inverted = check_inversion_needed(zmap, orientation)
    if inverted:
        zmap = zmap * -1

    segments = segment_zmap(zmap, pixel_size_um, orientation, segment_width_um)
    results = []

    for seg_idx, segment in enumerate(segments):
        profile_raw = extract_profile(segment, orientation)
        profile = (gaussian_filter1d(profile_raw, sigma=params.gaussian_sigma)
                   if params.gaussian_sigma > 0 else profile_raw)
        try:
            result = _qfactor_from_profile(profile, pixel_size_um, params)
            result["segment_id"] = seg_idx
            result["orientation"] = orientation
            result["inverted"] = inverted
            results.append(result)
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_qfactor(
    filepath,
    segment_width_um: float = 1.0,
    params: QFactorParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """
    Compute Q-factors for all slices of a single z-map image.

    The image is sliced into strips of `segment_width_um` width perpendicular
    to the trench direction. Each strip produces one row in the output.

    Parameters
    ----------
    filepath : str or Path
        Path to the OME-TIFF z-map file.
    segment_width_um : float
        Width of each slice in micrometers. Controls the trade-off between
        spatial resolution (small value) and profile stability (large value).
        Default is 1.0 µm.
    params : QFactorParams, optional
        Calculation parameters (baseline, smoothing, window size).
        Defaults to the optimal configuration from the study.

    Returns
    -------
    pd.DataFrame
        One row per slice. Columns:
            filename        : source file name
            slice_id        : integer index of the slice within the image
            position_um     : physical start position of the slice (µm)
            q_factor        : Q-factor (NaN if calculation failed)
            A               : ablated area (nm·µm)
            R               : redeposited area (nm·µm)
            baseline        : estimated baseline height (nm)
            sigma_shoulder  : shoulder roughness (nm)
            orientation     : 'horizontal' or 'vertical'
            inverted        : True if z-axis was flipped
            trench_start    : trench region start index (pixels)
            trench_end      : trench region end index (pixels)
            pixel_size_um   : pixel size of the source image (µm/pixel)
        Returns empty DataFrame if pixel size metadata is missing.
    """
    filepath = Path(filepath)
    pixel_size_um = extract_pixel_size(filepath)
    if pixel_size_um is None:
        return pd.DataFrame()

    zmap = load_zmap(filepath)
    orientation = detect_trench_orientation(zmap)
    inverted = check_inversion_needed(zmap, orientation)
    if inverted:
        zmap = zmap * -1

    segments = segment_zmap(zmap, pixel_size_um, orientation, segment_width_um)

    rows = []
    for slice_id, segment in enumerate(segments):
        position_um = slice_id * segment_width_um
        profile_raw = extract_profile(segment, orientation)
        profile = (gaussian_filter1d(profile_raw, sigma=params.gaussian_sigma)
                   if params.gaussian_sigma > 0 else profile_raw)
        try:
            result = _qfactor_from_profile(profile, pixel_size_um, params)
        except Exception:
            result = {
                "q_factor": np.nan, "A": np.nan, "R": np.nan,
                "baseline": np.nan, "sigma_shoulder": np.nan,
                "trench_start": np.nan, "trench_end": np.nan,
            }

        rows.append({
            "filename":       filepath.name,
            "slice_id":       slice_id,
            "position_um":    position_um,
            "q_factor":       result["q_factor"],
            "A":              result["A"],
            "R":              result["R"],
            "baseline":       result["baseline"],
            "sigma_shoulder": result["sigma_shoulder"],
            "orientation":    orientation,
            "inverted":       inverted,
            "trench_start":   result["trench_start"],
            "trench_end":     result["trench_end"],
            "pixel_size_um":  pixel_size_um,
        })

    return pd.DataFrame(rows)


def process_dataset(
    zmap_files: list[tuple],
    segment_width_um: float = 1.0,
    params: QFactorParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """
    Batch-process a list of z-map files.

    Calls compute_qfactor() for each file and concatenates the results.
    Returns all slices from all files — no filtering applied.

    Parameters
    ----------
    zmap_files : list of (Path, str)
        Each tuple is (directory_path, filename), matching the format
        produced by the data loading cell in the notebook.
    segment_width_um : float
        Slice width in micrometers, passed to compute_qfactor().
    params : QFactorParams, optional
        Calculation parameters, passed to compute_qfactor().

    Returns
    -------
    pd.DataFrame
        All slices from all files concatenated. Same columns as
        compute_qfactor(). Join with metadata on 'filename'.
    """
    frames = []

    for zmap_path, zmap_file in tqdm(zmap_files, desc="Processing z-maps"):
        filepath = Path(zmap_path) / zmap_file
        df = compute_qfactor(filepath, segment_width_um, params)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
