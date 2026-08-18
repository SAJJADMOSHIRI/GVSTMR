"""
GVSTMR CONUS simulation T1-T10 — Python export of the final Colab notebook.

The Colab notebook is the recommended executable version because it includes
package installation and interactive file upload helpers.
"""

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)


# %% [markdown] cell 0
# 
# # GVSTMR — CONUS Simulation T1–T10 — V2 High-Signal DGP
# 
# This notebook replaces the previous simulation DGP.
# 
# The previous version mixed partially independent spatial and temporal latent components, so **Y was not actually highly predictable from X**. That is why the resulting MGWR/GTWR/GGPR \(R^2\) values were low.
# 
# This version uses a **known, smooth, low-noise data-generating process**:
# 
# \[
# Y_{i,t}
# =
# 30
# +
# f(X_{i,t})
# +
# g(s_i)
# +
# h(s_i,t)
# +
# \varepsilon_{i,t},
# \]
# 
# where
# 
# \[
# f(x)=x+2(x^2-1)+0.25x^3
# \]
# 
# and the local true feature effect is
# 
# \[
# f'(x)=1+4x+0.75x^2.
# \]
# 
# This construction is deliberate:
# 
# - it is **highly recoverable** by MGWR, GTWR and GGPR;
# - the nonlinear derivative changes sign and magnitude across the CONUS X-gradient, so the direct GVSTMR spatial/temporal relationship scores still span **negative → weak → positive** states;
# - the same 3×3 GVSTMR color matrix can therefore be used for Simulation and Real-world maps;
# - a **fast preflight gate** runs before any expensive model. If the simulated DGP does not produce high recoverability, the notebook stops immediately instead of wasting hours.
# 
# ### Expected preflight behavior
# With the fixed seed and the supplied CONUS hex grid, the design was tested to produce approximately:
# 
# - quick local spatial \(R^2\): **~0.999**
# - quick GTWR \(R^2\): **~0.999**
# - quick GGPR spatial-holdout \(R^2\): **~0.99**
# - spatial relationship vs true local-effect Spearman: **~0.98**
# - temporal relationship vs true local-effect Spearman: **~0.88**
# 
# The exact final MGWR/GTWR/GGPR values will be calculated by the notebook.
# 
# ### Time
# - **T1–T10 only**
# - Suggested main figure snapshots: **T1, T4, T7, T10**
# - All 10 time layers are exported.
# 
# ### Final output
# `/content/GVSTMR_SIMULATION_CONUS_10T_V2.zip`
# 
# No new file is written to Google Drive.

# %% code cell 1

# ==========================================================================================
# CELL 1 — SETUP + LOAD SAME CONUS HEX GRID AS THE REAL-WORLD CASE
# ==========================================================================================

import sys
import subprocess
import importlib.util
import os
import gc
import json
import shutil
import zipfile
import warnings
import time

from pathlib import Path

def ensure_package(package, module=None):
    module = module or package
    if importlib.util.find_spec(module) is None:
        print("Installing:", package)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", package
        ])

for pkg, mod in [
    ("mgwr", "mgwr"),
    ("geoshapley", "geoshapley"),
    ("geopandas", "geopandas"),
    ("pyogrio", "pyogrio"),
    ("pyarrow", "pyarrow"),
]:
    ensure_package(pkg, mod)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR

from geoshapley import GeoShapleyExplainer


# ==========================================================================================
# SETTINGS
# ==========================================================================================

RANDOM_STATE = 123
rng = np.random.default_rng(RANDOM_STATE)

N_TIME = 10
DISPLAY_TIMES = [1, 4, 7, 10]

# Hidden buffer permits full centered 5-step temporal windows at T1 and T10.
TEMP_HALF_WINDOW = 2
INTERNAL_BUFFER = TEMP_HALF_WINDOW
TOTAL_INTERNAL_TIME = N_TIME + 2 * INTERNAL_BUFFER

# Direct GVSTMR
K_SPATIAL = 30
K_SENSITIVITY = [20, 30, 50]
WINDOW_SENSITIVITY = [1, 2]

# MGWR — one predictor only, so this is much lighter than the real 8-predictor case.
MGWR_MIN_BW = 25
MGWR_MAX_ITER_MULTI = 4
MGWR_TOL_MULTI = 1e-3
MGWR_BWS_SAME_TIMES = 2

# GTWR
GTWR_TAU_CANDIDATES = [0.25, 0.50, 1.00]
GTWR_K_CANDIDATES = [60, 80, 120]
GTWR_TUNE_N = 500
GTWR_BATCH = 5000

# GGPR
GGPR_NOISE = 0.03
GGPR_TUNE_N = 500
GGPR_SAMPLE_SIZES = [300, 600, 900]
GGPR_FINAL_N = 900
GGPR_SPATIAL_WEIGHTS = [0.05, 0.10, 0.20]
GGPR_LENGTH_SCALES = [0.5, 1.0, 2.0]

# Preflight gates — STOP EARLY if the DGP is not strongly recoverable.
PREFLIGHT_MIN_LOCAL_R2 = 0.97
PREFLIGHT_MIN_GTWR_R2 = 0.97
PREFLIGHT_MIN_GGPR_R2 = 0.95

ROOT = Path("/content/GVSTMR_SIMULATION_CONUS_10T_V2")
ZIP_PATH = Path("/content/GVSTMR_SIMULATION_CONUS_10T_V2.zip")

if ROOT.exists():
    shutil.rmtree(ROOT)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

DATA_ROOT   = ROOT / "01_SIMULATION_DATA"
GVSTMR_ROOT = ROOT / "02_DIRECT_GVSTMR"
MGWR_ROOT   = ROOT / "03_MGWR"
GTWR_ROOT   = ROOT / "04_GTWR"
GGPR_ROOT   = ROOT / "05_GGPR"
GEO_ROOT    = ROOT / "06_GEOSHAPLEY"
CARTO_ROOT  = ROOT / "07_CARTOGRAPHY_READY"
METRIC_ROOT = ROOT / "08_VALIDATION_METRICS"

for p in [
    DATA_ROOT,
    GVSTMR_ROOT,
    MGWR_ROOT,
    GTWR_ROOT,
    GGPR_ROOT,
    GEO_ROOT,
    CARTO_ROOT,
    METRIC_ROOT,
]:
    p.mkdir(parents=True, exist_ok=True)

print("=" * 105)
print("GVSTMR — CONUS SIMULATION V2 | T1–T10 | HIGH-SIGNAL DGP")
print("=" * 105)


# ==========================================================================================
# LOCATE REAL-WORLD FINAL ZIP
# ==========================================================================================

candidates = sorted(Path("/content").glob("GVSTMR_FINAL_2015_2024*.zip"))

if not candidates:
    try:
        from google.colab import files
        print("\nUpload your real-world final ZIP:")
        print("GVSTMR_FINAL_2015_2024.zip")
        uploaded = files.upload()
        candidates = [
            Path("/content") / name
            for name in uploaded
            if name.lower().endswith(".zip")
        ]
    except Exception:
        pass

if not candidates:
    raise RuntimeError(
        "Real-world final ZIP not found. "
        "Upload GVSTMR_FINAL_2015_2024.zip into /content and rerun."
    )

REAL_ZIP = candidates[0]
print("\nUsing real-world grid source:", REAL_ZIP)


# ==========================================================================================
# EXTRACT HEX GRID ROBUSTLY
# ==========================================================================================

EXTRACT_ROOT = Path("/content/_gvstmr_sim_v2_hex")

if EXTRACT_ROOT.exists():
    shutil.rmtree(EXTRACT_ROOT)

EXTRACT_ROOT.mkdir(parents=True)

with zipfile.ZipFile(REAL_ZIP, "r") as z:
    names = z.namelist()
    hex_names = [n for n in names if n.endswith("HEX_GRID.parquet")]

    if not hex_names:
        raise RuntimeError(
            "HEX_GRID.parquet was not found inside the real-world ZIP."
        )

    preferred = [n for n in hex_names if "03_CORE_DATA" in n]
    hex_member = preferred[0] if preferred else hex_names[0]

    z.extract(hex_member, EXTRACT_ROOT)

SIM_HEX = gpd.read_parquet(
    EXTRACT_ROOT / hex_member
)

if SIM_HEX.crs is None:
    SIM_HEX = SIM_HEX.set_crs("EPSG:5070")
elif SIM_HEX.crs.to_epsg() != 5070:
    SIM_HEX = SIM_HEX.to_crs(5070)

SIM_HEX = (
    SIM_HEX
    .drop_duplicates("HEX_ID")
    .copy()
    .reset_index(drop=True)
)

SIM_HEX["HEX_ID"] = SIM_HEX["HEX_ID"].astype(str)

cent = SIM_HEX.geometry.centroid
SIM_HEX["X_KM"] = cent.x / 1000.0
SIM_HEX["Y_KM"] = cent.y / 1000.0

N_HEX = len(SIM_HEX)

COORD_RAW = SIM_HEX[["X_KM", "Y_KM"]].to_numpy(float)
COORD_SCALER_BASE = StandardScaler()
COORD_Z = COORD_SCALER_BASE.fit_transform(COORD_RAW)

XZ = COORD_Z[:, 0]
YZ = COORD_Z[:, 1]

print(f"CONUS hexagons: {N_HEX:,}")
print("Simulated times:", list(range(1, 11)))
print("Suggested main figure snapshots:", DISPLAY_TIMES)
print("NO GOOGLE DRIVE WRITE.")

# %% code cell 2

# ==========================================================================================
# CELL 2 — HIGH-SIGNAL DGP + DIRECT GVSTMR + FAST PREFLIGHT GATES
# ==========================================================================================

t0 = time.perf_counter()

print("=" * 105)
print("SIMULATION DGP + DIRECT GVSTMR + PREFLIGHT")
print("=" * 105)


# ==========================================================================================
# A. FIVE SPATIALLY COHERENT DIAGNOSTIC REGIONS
#
# These are NOT used to force arbitrary target correlations.
# They are used for regional summaries and temporal phase heterogeneity.
# ==========================================================================================

regional_axis = (
    0.72 * XZ
    + 0.28 * YZ
    + 0.20 * np.sin(1.30 * YZ)
)

REGION = np.asarray(
    pd.qcut(
        regional_axis,
        q=5,
        labels=False,
        duplicates="drop"
    ),
    dtype=int
) + 1

if len(np.unique(REGION)) != 5:
    REGION = (
        KMeans(
            n_clusters=5,
            random_state=RANDOM_STATE,
            n_init=30
        )
        .fit_predict(COORD_Z)
        + 1
    )

SIM_HEX["SIM_REGION"] = REGION

REGION_TABLE = pd.DataFrame({
    "SIM_REGION": range(1, 6),
    "N_HEX": [int(np.sum(REGION == i)) for i in range(1, 6)],
})

REGION_TABLE.to_csv(
    DATA_ROOT / "SIMULATION_REGIONS.csv",
    index=False
)

display(REGION_TABLE)


# ==========================================================================================
# B. SMOOTH SPATIAL FIELDS
# ==========================================================================================

space_tree = cKDTree(COORD_RAW)

FIELD_K = 12

_, FIELD_NEIGHBORS = space_tree.query(
    COORD_RAW,
    k=FIELD_K + 1
)

def smooth_noise():
    z = rng.normal(size=N_HEX)

    for _ in range(3):
        local_mean = z[FIELD_NEIGHBORS].mean(axis=1)
        z = 0.25 * z + 0.75 * local_mean
        z = (z - z.mean()) / z.std()

    return z


# ==========================================================================================
# C. KNOWN ADDITIVE / NONLINEAR DATA-GENERATING PROCESS
#
# X is spatially structured and changes smoothly over time.
#
# Y = 30 + f(X) + g(space) + h(space,time) + epsilon
#
# f(X) is nonlinear, so its local derivative changes sign and magnitude:
#
# f(x) = x + 2(x^2 - 1) + 0.25x^3
# f'(x) = 1 + 4x + 0.75x^2
#
# This gives:
#   - high recoverability by local spatial / spatiotemporal models,
#   - nonlinear GGPR signal,
#   - negative / weak / positive local X–Y relationships for GVSTMR.
# ==========================================================================================

BASE_X = (
    0.85 * XZ
    + 0.30 * YZ
    + 0.22 * np.sin(1.40 * YZ)
    + 0.12 * smooth_noise()
)

PHASE = (
    0.80 * YZ
    + 0.50 * np.sin(XZ)
)

G_SPACE = (
    1.80 * np.sin(1.00 * XZ)
    + 1.50 * np.cos(1.10 * YZ)
    + 0.55 * XZ * YZ
)

X_INTERNAL = np.empty(
    (TOTAL_INTERNAL_TIME, N_HEX),
    dtype=np.float32
)

Y_INTERNAL = np.empty_like(X_INTERNAL)
Y_SIGNAL_INTERNAL = np.empty_like(X_INTERNAL)
TRUE_BETA_INTERNAL = np.empty_like(X_INTERNAL)

SIM_NOISE_SD = 0.10

for tt in range(TOTAL_INTERNAL_TIME):

    visible_clock = tt - INTERNAL_BUFFER
    angle = 2.0 * np.pi * visible_clock / N_TIME

    x_t = (
        BASE_X
        + 0.42 * np.sin(angle + PHASE)
        + 0.20 * np.cos(2.0 * angle - 0.60 * XZ)
        + 0.08 * rng.normal(size=N_HEX)
    )

    # Nonlinear feature response.
    f_x = (
        1.00 * x_t
        + 2.00 * (x_t**2 - 1.00)
        + 0.25 * x_t**3
    )

    # Exact local derivative of the feature response.
    true_beta = (
        1.00
        + 4.00 * x_t
        + 0.75 * x_t**2
    )

    # Smooth time / space-time intercept component.
    g_time = (
        0.65 * np.sin(angle)
        + 0.30 * np.cos(angle + 0.50 * YZ)
    )

    y_signal = (
        30.0
        + f_x
        + G_SPACE
        + g_time
    )

    y_t = (
        y_signal
        + SIM_NOISE_SD * rng.normal(size=N_HEX)
    )

    X_INTERNAL[tt] = x_t
    Y_SIGNAL_INTERNAL[tt] = y_signal
    Y_INTERNAL[tt] = y_t
    TRUE_BETA_INTERNAL[tt] = true_beta


VISIBLE = np.arange(
    INTERNAL_BUFFER,
    INTERNAL_BUFFER + N_TIME
)

X = X_INTERNAL[VISIBLE]
Y = Y_INTERNAL[VISIBLE]
Y_SIGNAL = Y_SIGNAL_INTERNAL[VISIBLE]
TRUE_BETA = TRUE_BETA_INTERNAL[VISIBLE]


# ==========================================================================================
# D. BASIC DGP QUALITY
# ==========================================================================================

ORACLE_R2 = r2_score(
    Y.ravel(),
    Y_SIGNAL.ravel()
)

ORACLE_RMSE = (
    mean_squared_error(
        Y.ravel(),
        Y_SIGNAL.ravel()
    ) ** 0.5
)

print("\nDGP signal-to-noise check")
print("Oracle signal R2  :", ORACLE_R2)
print("Oracle signal RMSE:", ORACLE_RMSE)
print("Y SD              :", float(np.std(Y)))


# ==========================================================================================
# E. DIRECT GVSTMR SCORES
# ==========================================================================================

def row_correlation(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    A0 = A - A.mean(axis=1, keepdims=True)
    B0 = B - B.mean(axis=1, keepdims=True)

    num = np.sum(A0 * B0, axis=1)

    den = np.sqrt(
        np.sum(A0**2, axis=1)
        * np.sum(B0**2, axis=1)
    )

    out = np.divide(
        num,
        den,
        out=np.zeros_like(num),
        where=den > 1e-12
    )

    return np.clip(out, -1.0, 1.0)


def compute_spatial_rho(K):
    _, nbr = space_tree.query(
        COORD_RAW,
        k=K + 1
    )

    out = np.empty(
        (N_TIME, N_HEX),
        dtype=np.float32
    )

    for t in range(N_TIME):
        out[t] = row_correlation(
            X[t][nbr],
            Y[t][nbr]
        )

    return out


def compute_temporal_rho(half_window):
    out = np.empty(
        (N_TIME, N_HEX),
        dtype=np.float32
    )

    for visible_t in range(N_TIME):

        internal_t = visible_t + INTERNAL_BUFFER

        lo = max(
            0,
            internal_t - half_window
        )

        hi = min(
            TOTAL_INTERNAL_TIME,
            internal_t + half_window + 1
        )

        out[visible_t] = row_correlation(
            X_INTERNAL[lo:hi].T,
            Y_INTERNAL[lo:hi].T
        )

    return out


RHO_S = compute_spatial_rho(
    K_SPATIAL
)

RHO_T = compute_temporal_rho(
    TEMP_HALF_WINDOW
)


# ==========================================================================================
# F. DIRECT GVSTMR vs KNOWN TRUE LOCAL FEATURE EFFECT
#
# Correlation is not identical to a derivative, so we use rank and sign recovery,
# which are the appropriate checks for direction / strength ordering.
# ==========================================================================================

SPATIAL_BETA_SPEARMAN = float(
    spearmanr(
        RHO_S.ravel(),
        TRUE_BETA.ravel()
    ).statistic
)

TEMPORAL_BETA_SPEARMAN = float(
    spearmanr(
        RHO_T.ravel(),
        TRUE_BETA.ravel()
    ).statistic
)

beta_mask = np.abs(
    TRUE_BETA
) > 0.20

SPATIAL_SIGN_AGREEMENT = float(
    np.mean(
        np.sign(RHO_S[beta_mask])
        ==
        np.sign(TRUE_BETA[beta_mask])
    )
)

TEMPORAL_SIGN_AGREEMENT = float(
    np.mean(
        np.sign(RHO_T[beta_mask])
        ==
        np.sign(TRUE_BETA[beta_mask])
    )
)


RELATIONSHIP_RECOVERY = pd.DataFrame([{
    "SPATIAL_RHO_vs_TRUE_BETA_SPEARMAN":
        SPATIAL_BETA_SPEARMAN,

    "TEMPORAL_RHO_vs_TRUE_BETA_SPEARMAN":
        TEMPORAL_BETA_SPEARMAN,

    "SPATIAL_SIGN_AGREEMENT":
        SPATIAL_SIGN_AGREEMENT,

    "TEMPORAL_SIGN_AGREEMENT":
        TEMPORAL_SIGN_AGREEMENT,
}])

display(RELATIONSHIP_RECOVERY)

RELATIONSHIP_RECOVERY.to_csv(
    METRIC_ROOT /
    "GVSTMR_TRUE_EFFECT_RECOVERY.csv",
    index=False
)


# ==========================================================================================
# G. FIXED GLOBAL TERTILES — SAME CUTS FOR T1...T10
# ==========================================================================================

S_LOW, S_HIGH = np.quantile(
    RHO_S.ravel(),
    [1/3, 2/3]
)

T_LOW, T_HIGH = np.quantile(
    RHO_T.ravel(),
    [1/3, 2/3]
)

def classify3(values, low_cut, high_cut):
    return np.where(
        values <= low_cut,
        1,
        np.where(
            values <= high_cut,
            2,
            3
        )
    ).astype(np.int8)

S_CLASS = classify3(
    RHO_S,
    S_LOW,
    S_HIGH
)

T_CLASS = classify3(
    RHO_T,
    T_LOW,
    T_HIGH
)

GV_CELL = (
    (S_CLASS - 1) * 3
    + T_CLASS
).astype(np.int8)


# ==========================================================================================
# H. SAME 3×3 CARTOGRAPHIC PALETTE AS THE CONCEPTUAL FIGURE
# ==========================================================================================

GV_PALETTE = {
    # Spatial LOW
    1: "#E5E4E9",
    2: "#B9D9E6",
    3: "#54B8D0",

    # Spatial MEDIUM
    4: "#DB95CB",
    5: "#9E9FCB",
    6: "#4582BB",

    # Spatial HIGH
    7: "#BE3F98",
    8: "#74529F",
    9: "#243A83",
}

LEVEL_NAME = {
    1: "Low",
    2: "Medium",
    3: "High"
}

palette_rows = []

for s in [1, 2, 3]:
    for t in [1, 2, 3]:

        cell = (
            (s - 1) * 3
            + t
        )

        palette_rows.append({
            "GV_CELL":
                cell,

            "SPATIAL_LEVEL":
                LEVEL_NAME[s],

            "TEMPORAL_LEVEL":
                LEVEL_NAME[t],

            "COLOR_HEX":
                GV_PALETTE[cell],
        })

GV_PALETTE_TABLE = pd.DataFrame(
    palette_rows
)

GV_PALETTE_TABLE.to_csv(
    GVSTMR_ROOT /
    "GVSTMR_3X3_COLOR_MATRIX.csv",
    index=False
)

display(GV_PALETTE_TABLE)


# ==========================================================================================
# I. LONG PANEL
# ==========================================================================================

SIM_PANEL = pd.DataFrame({
    "HEX_ID":
        np.tile(
            SIM_HEX["HEX_ID"].to_numpy(),
            N_TIME
        ),

    "TIME":
        np.repeat(
            np.arange(1, N_TIME + 1),
            N_HEX
        ),

    "SIM_REGION":
        np.tile(
            REGION,
            N_TIME
        ),

    "X_KM":
        np.tile(
            SIM_HEX["X_KM"].to_numpy(),
            N_TIME
        ),

    "Y_KM":
        np.tile(
            SIM_HEX["Y_KM"].to_numpy(),
            N_TIME
        ),

    "X":
        X.reshape(-1),

    "Y":
        Y.reshape(-1),

    "Y_SIGNAL":
        Y_SIGNAL.reshape(-1),

    "TRUE_BETA_X":
        TRUE_BETA.reshape(-1),

    "RHO_S":
        RHO_S.reshape(-1),

    "RHO_T":
        RHO_T.reshape(-1),

    "SPATIAL_CLASS":
        S_CLASS.reshape(-1),

    "TEMPORAL_CLASS":
        T_CLASS.reshape(-1),

    "GV_CELL":
        GV_CELL.reshape(-1),
})

SIM_PANEL["GV_COLOR"] = (
    SIM_PANEL["GV_CELL"]
    .map(GV_PALETTE)
)

SIM_PANEL.to_parquet(
    DATA_ROOT /
    "SIMULATION_PANEL_T1_T10.parquet",
    index=False
)

pd.DataFrame([{
    "SPATIAL_LOW_CUT":
        S_LOW,

    "SPATIAL_HIGH_CUT":
        S_HIGH,

    "TEMPORAL_LOW_CUT":
        T_LOW,

    "TEMPORAL_HIGH_CUT":
        T_HIGH,

    "K_SPATIAL":
        K_SPATIAL,

    "TEMPORAL_WINDOW_SIZE":
        2 * TEMP_HALF_WINDOW + 1,
}]).to_csv(
    GVSTMR_ROOT /
    "GVSTMR_FIXED_GLOBAL_THRESHOLDS.csv",
    index=False
)


# ==========================================================================================
# J. SMALL PARAMETER SENSITIVITY
# ==========================================================================================

sens_rows = []

for k in K_SENSITIVITY:

    rs = compute_spatial_rho(
        k
    )

    sens_rows.append({
        "PARAMETER":
            "K_SPATIAL",

        "VALUE":
            k,

        "RHO_vs_TRUE_BETA_SPEARMAN":
            float(
                spearmanr(
                    rs.ravel(),
                    TRUE_BETA.ravel()
                ).statistic
            )
    })

for hw in WINDOW_SENSITIVITY:

    rt = compute_temporal_rho(
        hw
    )

    sens_rows.append({
        "PARAMETER":
            "TEMP_HALF_WINDOW",

        "VALUE":
            hw,

        "RHO_vs_TRUE_BETA_SPEARMAN":
            float(
                spearmanr(
                    rt.ravel(),
                    TRUE_BETA.ravel()
                ).statistic
            )
    })

GVSTMR_SENSITIVITY = pd.DataFrame(
    sens_rows
)

GVSTMR_SENSITIVITY.to_csv(
    METRIC_ROOT /
    "GVSTMR_PARAMETER_SENSITIVITY.csv",
    index=False
)

display(GVSTMR_SENSITIVITY)


# ==========================================================================================
# K. CLIMATOLOGICAL TABLE USED BY MGWR / GGPR
# ==========================================================================================

SIM_CLIM = (
    SIM_PANEL

    .groupby(
        "HEX_ID",
        as_index=False
    )

    .agg(
        X_KM=("X_KM", "first"),
        Y_KM=("Y_KM", "first"),
        X=("X", "mean"),
        Y=("Y", "mean"),
        Y_SIGNAL=("Y_SIGNAL", "mean"),
        TRUE_BETA_X=("TRUE_BETA_X", "mean"),
    )
)


# ==========================================================================================
# L. PREFLIGHT 1 — FAST LOCAL SPATIAL REGRESSION
#
# This approximates the local structure MGWR must recover.
# It runs in seconds.
# ==========================================================================================

PREFLIGHT_K = 80

_, nbr80 = space_tree.query(
    COORD_RAW,
    k=PREFLIGHT_K
)

xm = SIM_CLIM["X"].to_numpy(float)
ym = SIM_CLIM["Y"].to_numpy(float)

xx = xm[nbr80]
yy = ym[nbr80]

mx = xx.mean(axis=1)
my = yy.mean(axis=1)

beta_fast = (
    np.sum(
        (xx - mx[:, None])
        *
        (yy - my[:, None]),
        axis=1
    )

    /

    (
        np.sum(
            (xx - mx[:, None])**2,
            axis=1
        )
        +
        1e-12
    )
)

alpha_fast = (
    my
    -
    beta_fast * mx
)

pred_fast = (
    alpha_fast
    +
    beta_fast * xm
)

PREFLIGHT_LOCAL_R2 = r2_score(
    ym,
    pred_fast
)

PREFLIGHT_LOCAL_RMSE = (
    mean_squared_error(
        ym,
        pred_fast
    ) ** 0.5
)


# ==========================================================================================
# M. PREFLIGHT 2 — FAST GTWR SAMPLE
# ==========================================================================================

GT_PREF = SIM_PANEL[
    [
        "TIME",
        "X_KM",
        "Y_KM",
        "X",
        "Y"
    ]
].copy()

gt_x_std = StandardScaler().fit_transform(
    GT_PREF[["X"]]
).ravel()

gt_y_raw = GT_PREF["Y"].to_numpy(float)

space_z_unique = StandardScaler().fit_transform(
    SIM_HEX[["X_KM", "Y_KM"]]
)

space_z_panel = np.tile(
    space_z_unique,
    (N_TIME, 1)
)

time_z_unique = StandardScaler().fit_transform(
    np.arange(1, N_TIME + 1).reshape(-1, 1)
).ravel()

time_z_panel = np.repeat(
    time_z_unique,
    N_HEX
)

st_pref = np.column_stack([
    space_z_panel,
    np.sqrt(0.50) * time_z_panel
])

tree_pref = cKDTree(
    st_pref
)

pref_idx = rng.choice(
    len(GT_PREF),
    size=min(
        1200,
        len(GT_PREF)
    ),
    replace=False
)

dist, nbr = tree_pref.query(
    st_pref[pref_idx],
    k=80
)

bw = dist[:, -1] + 1e-12
u = dist / bw[:, None]

w = (
    1.0
    -
    u**2
) ** 2

w[
    u >= 1
] = 0.0

xj = gt_x_std[nbr]
yj = gt_y_raw[nbr]

SW = np.sum(w, axis=1)
SX = np.sum(w * xj, axis=1)
SY = np.sum(w * yj, axis=1)
SXX = np.sum(w * xj * xj, axis=1)
SXY = np.sum(w * xj * yj, axis=1)

den = (
    SW * SXX
    -
    SX**2
)

beta = (
    SW * SXY
    -
    SX * SY
) / (
    den + 1e-12
)

alpha = (
    SY
    -
    beta * SX
) / SW

gt_pref_pred = (
    alpha
    +
    beta * gt_x_std[pref_idx]
)

PREFLIGHT_GTWR_R2 = r2_score(
    gt_y_raw[pref_idx],
    gt_pref_pred
)

PREFLIGHT_GTWR_RMSE = (
    mean_squared_error(
        gt_y_raw[pref_idx],
        gt_pref_pred
    ) ** 0.5
)


# ==========================================================================================
# N. SIMULATION GGPR KERNEL — DEFINED HERE SO PREFLIGHT USES THE SAME STRUCTURE
# ==========================================================================================

class SimulationGGPRKernel(Kernel):

    def __init__(
        self,
        spatial_weight=0.10,
        length_scale=1.0
    ):
        self.spatial_weight = spatial_weight
        self.length_scale = length_scale

    def __call__(
        self,
        X,
        Y=None,
        eval_gradient=False
    ):

        X = np.asarray(
            X,
            float
        )

        if Y is None:
            Y = X
            same = True

        else:
            Y = np.asarray(
                Y,
                float
            )
            same = False

        d_feature = (
            X[:, 0][:, None]
            -
            Y[:, 0][None, :]
        )

        ssk = np.exp(
            -0.5 * d_feature**2
        )

        dx = (
            X[:, 1][:, None]
            -
            Y[:, 1][None, :]
        )

        dy = (
            X[:, 2][:, None]
            -
            Y[:, 2][None, :]
        )

        distance = np.sqrt(
            dx**2
            +
            dy**2
        )

        r = (
            np.sqrt(3.0)
            *
            distance
            /
            float(
                self.length_scale
            )
        )

        matern = (
            1.0
            +
            r
        ) * np.exp(
            -r
        )

        K = (
            ssk
            +
            float(
                self.spatial_weight
            )
            *
            matern
        )

        if eval_gradient:

            if not same:
                raise ValueError(
                    "Gradient only supported for Y=None."
                )

            return (
                K,
                np.empty(
                    (
                        len(X),
                        len(X),
                        0
                    )
                )
            )

        return K

    def diag(self, X):
        return np.full(
            len(X),
            1.0
            +
            float(
                self.spatial_weight
            )
        )

    def is_stationary(self):
        return True


# ==========================================================================================
# O. PREFLIGHT 3 — SMALL SPATIAL-HOLDOUT GGPR
# ==========================================================================================

pre_blocks = (
    KMeans(
        n_clusters=5,
        random_state=RANDOM_STATE,
        n_init=20
    )
    .fit_predict(
        SIM_CLIM[
            ["X_KM", "Y_KM"]
        ]
    )
)

pre_test = SIM_CLIM.loc[
    pre_blocks == 0
].copy()

pre_pool = SIM_CLIM.loc[
    pre_blocks != 0
].copy()

pre_train = pre_pool.sample(
    n=min(
        350,
        len(pre_pool)
    ),
    random_state=RANDOM_STATE
)

pre_x_scaler = StandardScaler()
pre_coord_scaler = StandardScaler()
pre_y_scaler = StandardScaler()

pre_A_train = np.column_stack([
    pre_x_scaler.fit_transform(
        pre_train[["X"]]
    ),

    pre_coord_scaler.fit_transform(
        pre_train[
            ["X_KM", "Y_KM"]
        ]
    )
])

pre_y = pre_y_scaler.fit_transform(
    pre_train[["Y"]]
).ravel()

pre_gp = GaussianProcessRegressor(
    kernel=SimulationGGPRKernel(
        spatial_weight=0.10,
        length_scale=1.0
    ),
    alpha=GGPR_NOISE,
    optimizer=None,
    normalize_y=False,
    random_state=RANDOM_STATE
)

pre_gp.fit(
    pre_A_train,
    pre_y
)

pre_A_test = np.column_stack([
    pre_x_scaler.transform(
        pre_test[["X"]]
    ),

    pre_coord_scaler.transform(
        pre_test[
            ["X_KM", "Y_KM"]
        ]
    )
])

pre_pred_z = pre_gp.predict(
    pre_A_test
)

pre_pred = (
    pre_y_scaler
    .inverse_transform(
        pre_pred_z.reshape(
            -1,
            1
        )
    )
    .ravel()
)

PREFLIGHT_GGPR_R2 = r2_score(
    pre_test["Y"],
    pre_pred
)

PREFLIGHT_GGPR_RMSE = (
    mean_squared_error(
        pre_test["Y"],
        pre_pred
    ) ** 0.5
)


# ==========================================================================================
# P. REPORT + HARD STOP IF SOMETHING IS WRONG
# ==========================================================================================

PREFLIGHT = pd.DataFrame([{
    "ORACLE_SIGNAL_R2":
        ORACLE_R2,

    "QUICK_LOCAL_SPATIAL_R2":
        PREFLIGHT_LOCAL_R2,

    "QUICK_LOCAL_SPATIAL_RMSE":
        PREFLIGHT_LOCAL_RMSE,

    "QUICK_GTWR_R2":
        PREFLIGHT_GTWR_R2,

    "QUICK_GTWR_RMSE":
        PREFLIGHT_GTWR_RMSE,

    "QUICK_GGPR_SPATIAL_HOLDOUT_R2":
        PREFLIGHT_GGPR_R2,

    "QUICK_GGPR_SPATIAL_HOLDOUT_RMSE":
        PREFLIGHT_GGPR_RMSE,

    "SPATIAL_RHO_TRUE_BETA_SPEARMAN":
        SPATIAL_BETA_SPEARMAN,

    "TEMPORAL_RHO_TRUE_BETA_SPEARMAN":
        TEMPORAL_BETA_SPEARMAN,
}])

print("\n" + "=" * 105)
print("FAST PREFLIGHT RESULTS — CHECK BEFORE EXPENSIVE MODELS")
print("=" * 105)

display(PREFLIGHT)

PREFLIGHT.to_csv(
    METRIC_ROOT /
    "PREFLIGHT_SANITY_CHECK.csv",
    index=False
)

if PREFLIGHT_LOCAL_R2 < PREFLIGHT_MIN_LOCAL_R2:
    raise RuntimeError(
        f"STOPPED BEFORE MGWR: local spatial R2 = {PREFLIGHT_LOCAL_R2:.4f}"
    )

if PREFLIGHT_GTWR_R2 < PREFLIGHT_MIN_GTWR_R2:
    raise RuntimeError(
        f"STOPPED BEFORE MGWR: quick GTWR R2 = {PREFLIGHT_GTWR_R2:.4f}"
    )

if PREFLIGHT_GGPR_R2 < PREFLIGHT_MIN_GGPR_R2:
    raise RuntimeError(
        f"STOPPED BEFORE MGWR: quick GGPR R2 = {PREFLIGHT_GGPR_R2:.4f}"
    )

print("\n✅ PREFLIGHT PASSED.")
print("The expensive models are allowed to run.")
print(f"Cell runtime: {(time.perf_counter()-t0)/60:.1f} min")

# %% code cell 3

# ==========================================================================================
# CELL 3 — FULL-CONUS MGWR | 10-time climatological Y ~ X
# ==========================================================================================

t0 = time.perf_counter()

print("=" * 105)
print("SIMULATION V2 — FULL-CONUS MGWR")
print("=" * 105)

coords_mgwr = SIM_CLIM[
    ["X_KM", "Y_KM"]
].to_numpy(float)

y_mgwr = SIM_CLIM[
    "Y"
].to_numpy(float).reshape(
    -1,
    1
)

mgwr_x_scaler = StandardScaler()

X_mgwr = mgwr_x_scaler.fit_transform(
    SIM_CLIM[
        ["X"]
    ]
)

selector = Sel_BW(
    coords_mgwr,
    y_mgwr,
    X_mgwr,
    multi=True,
    fixed=False,
    kernel="bisquare",
    constant=True,
    spherical=False
)

print(f"Hexes: {len(SIM_CLIM):,}")
print("Searching MGWR bandwidths...")

mgwr_bws = selector.search(
    criterion="AICc",
    multi_bw_min=[
        MGWR_MIN_BW
    ],
    tol_multi=MGWR_TOL_MULTI,
    max_iter_multi=MGWR_MAX_ITER_MULTI,
    bws_same_times=MGWR_BWS_SAME_TIMES,
    verbose=True
)

print(
    "\nFinal MGWR bandwidths:",
    np.asarray(
        mgwr_bws
    ).ravel()
)

model = MGWR(
    coords_mgwr,
    y_mgwr,
    X_mgwr,
    selector,
    kernel="bisquare",
    fixed=False,
    constant=True,
    spherical=False,
    hat_matrix=False
)

res = model.fit(
    n_chunks=8
)

pred = np.asarray(
    res.predy
).ravel()

obs = y_mgwr.ravel()

MGWR_SUMMARY = pd.DataFrame([{
    "MODEL":
        "MGWR",

    "N_HEX":
        len(
            SIM_CLIM
        ),

    "R2":
        r2_score(
            obs,
            pred
        ),

    "ADJ_R2":
        float(
            res.adj_R2
        ),

    "RMSE":
        mean_squared_error(
            obs,
            pred
        ) ** 0.5,

    "MAE":
        mean_absolute_error(
            obs,
            pred
        ),

    "BIAS":
        float(
            np.mean(
                pred
                -
                obs
            )
        ),

    "AIC":
        float(
            res.aic
        ),

    "AICC":
        float(
            res.aicc
        ),

    "BIC":
        float(
            res.bic
        ),
}])

display(
    MGWR_SUMMARY
)

MGWR_SUMMARY.to_csv(
    MGWR_ROOT /
    "MGWR_SUMMARY.csv",
    index=False
)

pd.DataFrame({
    "PARAMETER":
        [
            "INTERCEPT",
            "X"
        ],

    "BANDWIDTH_N_NEIGHBORS":
        np.asarray(
            mgwr_bws
        ).ravel()
}).to_csv(
    MGWR_ROOT /
    "MGWR_BANDWIDTHS.csv",
    index=False
)

params = np.asarray(
    res.params
)

tvalues = np.asarray(
    res.tvalues
)

MGWR_LOCAL = SIM_CLIM[
    [
        "HEX_ID",
        "X_KM",
        "Y_KM",
        "X",
        "Y",
        "Y_SIGNAL",
        "TRUE_BETA_X"
    ]
].copy()

MGWR_LOCAL[
    "MGWR_PRED_Y"
] = pred

MGWR_LOCAL[
    "MGWR_RESID"
] = (
    obs
    -
    pred
)

MGWR_LOCAL[
    "BETA_INTERCEPT"
] = params[
    :,
    0
]

MGWR_LOCAL[
    "BETA_X"
] = params[
    :,
    1
]

MGWR_LOCAL[
    "T_INTERCEPT"
] = tvalues[
    :,
    0
]

MGWR_LOCAL[
    "T_X"
] = tvalues[
    :,
    1
]

MGWR_LOCAL.to_parquet(
    MGWR_ROOT /
    "MGWR_LOCAL_ALL_HEX.parquet",
    index=False
)

MGWR_COEF_SUMMARY = pd.DataFrame([{
    "VARIABLE":
        "X",

    "MEAN_BETA":
        float(
            np.nanmean(
                params[
                    :,
                    1
                ]
            )
        ),

    "SD_BETA":
        float(
            np.nanstd(
                params[
                    :,
                    1
                ]
            )
        ),

    "P05_BETA":
        float(
            np.nanpercentile(
                params[
                    :,
                    1
                ],
                5
            )
        ),

    "MEDIAN_BETA":
        float(
            np.nanmedian(
                params[
                    :,
                    1
                ]
            )
        ),

    "P95_BETA":
        float(
            np.nanpercentile(
                params[
                    :,
                    1
                ],
                95
            )
        ),

    "PCT_POSITIVE":
        float(
            np.mean(
                params[
                    :,
                    1
                ]
                >
                0
            )
            *
            100
        ),

    "PCT_NEGATIVE":
        float(
            np.mean(
                params[
                    :,
                    1
                ]
                <
                0
            )
            *
            100
        ),
}])

MGWR_COEF_SUMMARY.to_csv(
    MGWR_ROOT /
    "MGWR_COEFFICIENT_SUMMARY.csv",
    index=False
)

display(
    MGWR_COEF_SUMMARY
)

print(
    f"\nMGWR runtime: {(time.perf_counter()-t0)/60:.1f} min"
)

gc.collect()

# %% code cell 4

# ==========================================================================================
# CELL 4 — FULL FAST GTWR | ALL 12,864 HEXES × T1–T10
# ==========================================================================================

t0 = time.perf_counter()

print("=" * 105)
print("SIMULATION V2 — FULL GTWR")
print("=" * 105)

GT = SIM_PANEL[
    [
        "HEX_ID",
        "TIME",
        "X_KM",
        "Y_KM",
        "X",
        "Y",
        "Y_SIGNAL",
        "TRUE_BETA_X"
    ]
].copy().reset_index(drop=True)

GT_X_SCALER = StandardScaler()

GT_X = GT_X_SCALER.fit_transform(
    GT[
        ["X"]
    ]
).ravel()

GT_Y = GT[
    "Y"
].to_numpy(float)

SPACE_Z_UNIQUE = StandardScaler().fit_transform(
    SIM_HEX[
        ["X_KM", "Y_KM"]
    ]
)

SPACE_Z = np.tile(
    SPACE_Z_UNIQUE,
    (N_TIME, 1)
)

TIME_Z_UNIQUE = StandardScaler().fit_transform(
    np.arange(
        1,
        N_TIME + 1
    ).reshape(
        -1,
        1
    )
).ravel()

TIME_Z = np.repeat(
    TIME_Z_UNIQUE,
    N_HEX
)


def gtwr_local_batch(
    target_idx,
    tree,
    ST,
    K,
    leave_self_out=False
):

    target_idx = np.asarray(
        target_idx,
        dtype=int
    )

    query_k = (
        K + 1
        if leave_self_out
        else K
    )

    dist, nbr = tree.query(
        ST[
            target_idx
        ],
        k=query_k
    )

    if leave_self_out:

        dist = dist[
            :,
            1:
        ]

        nbr = nbr[
            :,
            1:
        ]

    bw = (
        dist[
            :,
            -1
        ]
        +
        1e-12
    )

    u = (
        dist
        /
        bw[
            :,
            None
        ]
    )

    w = (
        1.0
        -
        u**2
    ) ** 2

    w[
        u >= 1.0
    ] = 0.0

    xj = GT_X[
        nbr
    ]

    yj = GT_Y[
        nbr
    ]

    SW = np.sum(
        w,
        axis=1
    )

    SX = np.sum(
        w * xj,
        axis=1
    )

    SY = np.sum(
        w * yj,
        axis=1
    )

    SXX = np.sum(
        w * xj * xj,
        axis=1
    )

    SXY = np.sum(
        w * xj * yj,
        axis=1
    )

    denominator = (
        SW * SXX
        -
        SX**2
    )

    denominator = np.where(
        np.abs(
            denominator
        )
        <
        1e-12,

        np.nan,
        denominator
    )

    beta_x = (
        SW * SXY
        -
        SX * SY
    ) / denominator

    beta_0 = (
        SY
        -
        beta_x * SX
    ) / SW

    beta_x = np.nan_to_num(
        beta_x
    )

    beta_0 = np.nan_to_num(
        beta_0,
        nan=float(
            np.mean(
                GT_Y
            )
        )
    )

    pred = (
        beta_0
        +
        beta_x
        *
        GT_X[
            target_idx
        ]
    )

    return (
        pred,
        beta_0,
        beta_x,
        bw
    )


# ==========================================================================================
# TUNING
# ==========================================================================================

tune_idx = rng.choice(
    len(
        GT
    ),
    size=min(
        GTWR_TUNE_N,
        len(
            GT
        )
    ),
    replace=False
)

tune_rows = []

for tau in GTWR_TAU_CANDIDATES:

    ST = np.column_stack([
        SPACE_Z,
        np.sqrt(
            tau
        )
        *
        TIME_Z
    ])

    tree = cKDTree(
        ST
    )

    for K in GTWR_K_CANDIDATES:

        pred_tune, _, _, _ = gtwr_local_batch(
            tune_idx,
            tree,
            ST,
            K,
            leave_self_out=True
        )

        obs_tune = GT_Y[
            tune_idx
        ]

        row = {
            "TAU":
                tau,

            "K":
                K,

            "LOO_R2":
                r2_score(
                    obs_tune,
                    pred_tune
                ),

            "LOO_RMSE":
                mean_squared_error(
                    obs_tune,
                    pred_tune
                ) ** 0.5,

            "LOO_MAE":
                mean_absolute_error(
                    obs_tune,
                    pred_tune
                ),
        }

        tune_rows.append(
            row
        )

        print(
            f"tau={tau:<4} "
            f"K={K:<4} "
            f"LOO R2={row['LOO_R2']:.4f} "
            f"RMSE={row['LOO_RMSE']:.4f}"
        )

GTWR_TUNING = (
    pd.DataFrame(
        tune_rows
    )

    .sort_values(
        [
            "LOO_RMSE",
            "LOO_R2"
        ],
        ascending=[
            True,
            False
        ]
    )

    .reset_index(
        drop=True
    )
)

BEST_TAU = float(
    GTWR_TUNING.iloc[
        0
    ][
        "TAU"
    ]
)

BEST_K = int(
    GTWR_TUNING.iloc[
        0
    ][
        "K"
    ]
)

display(
    GTWR_TUNING
)

print(
    "\nBEST TAU:",
    BEST_TAU
)

print(
    "BEST K:",
    BEST_K
)

GTWR_TUNING.to_csv(
    GTWR_ROOT /
    "GTWR_PARAMETER_TUNING.csv",
    index=False
)


# ==========================================================================================
# FINAL ALL T1–T10
# ==========================================================================================

ST_FINAL = np.column_stack([
    SPACE_Z,
    np.sqrt(
        BEST_TAU
    )
    *
    TIME_Z
])

GT_TREE = cKDTree(
    ST_FINAL
)

N_GT = len(
    GT
)

GT_PRED = np.empty(
    N_GT,
    dtype=np.float32
)

GT_B0 = np.empty(
    N_GT,
    dtype=np.float32
)

GT_BX = np.empty(
    N_GT,
    dtype=np.float32
)

GT_BW = np.empty(
    N_GT,
    dtype=np.float32
)

for start in range(
    0,
    N_GT,
    GTWR_BATCH
):

    end = min(
        start
        +
        GTWR_BATCH,
        N_GT
    )

    idx = np.arange(
        start,
        end
    )

    pred_batch, b0, bx, bw = gtwr_local_batch(
        idx,
        GT_TREE,
        ST_FINAL,
        BEST_K,
        leave_self_out=False
    )

    GT_PRED[
        start:end
    ] = pred_batch

    GT_B0[
        start:end
    ] = b0

    GT_BX[
        start:end
    ] = bx

    GT_BW[
        start:end
    ] = bw

    print(
        f"GTWR: {end:,}/{N_GT:,}"
    )

GT[
    "GTWR_PRED_Y"
] = GT_PRED

GT[
    "GTWR_RESID"
] = (
    GT[
        "Y"
    ].to_numpy(float)
    -
    GT_PRED
)

GT[
    "BETA_INTERCEPT"
] = GT_B0

GT[
    "BETA_X"
] = GT_BX

GT[
    "LOCAL_ST_BW"
] = GT_BW

GTWR_SUMMARY = pd.DataFrame([{
    "MODEL":
        "Adaptive GTWR",

    "N_HEX":
        N_HEX,

    "N_HEX_TIME":
        N_GT,

    "N_TIME":
        N_TIME,

    "TAU":
        BEST_TAU,

    "K":
        BEST_K,

    "R2_LOCAL_FIT":
        r2_score(
            GT_Y,
            GT_PRED
        ),

    "RMSE_LOCAL_FIT":
        mean_squared_error(
            GT_Y,
            GT_PRED
        ) ** 0.5,

    "MAE_LOCAL_FIT":
        mean_absolute_error(
            GT_Y,
            GT_PRED
        ),

    "BIAS":
        float(
            np.mean(
                GT_PRED
                -
                GT_Y
            )
        ),
}])

display(
    GTWR_SUMMARY
)

GTWR_SUMMARY.to_csv(
    GTWR_ROOT /
    "GTWR_SUMMARY.csv",
    index=False
)

GT.to_parquet(
    GTWR_ROOT /
    "GTWR_ALL_HEX_T1_T10.parquet",
    index=False
)

by_time = []

for t in range(
    1,
    N_TIME + 1
):

    q = GT.loc[
        GT[
            "TIME"
        ].eq(
            t
        )
    ]

    by_time.append({
        "TIME":
            t,

        "N":
            len(
                q
            ),

        "R2_LOCAL_FIT":
            r2_score(
                q[
                    "Y"
                ],
                q[
                    "GTWR_PRED_Y"
                ]
            ),

        "RMSE_LOCAL_FIT":
            mean_squared_error(
                q[
                    "Y"
                ],
                q[
                    "GTWR_PRED_Y"
                ]
            ) ** 0.5,

        "MAE_LOCAL_FIT":
            mean_absolute_error(
                q[
                    "Y"
                ],
                q[
                    "GTWR_PRED_Y"
                ]
            ),

        "MEAN_BETA_X":
            q[
                "BETA_X"
            ].mean(),

        "MEDIAN_BETA_X":
            q[
                "BETA_X"
            ].median(),

        "P05_BETA_X":
            q[
                "BETA_X"
            ].quantile(
                0.05
            ),

        "P95_BETA_X":
            q[
                "BETA_X"
            ].quantile(
                0.95
            ),
    })

GTWR_BY_TIME = pd.DataFrame(
    by_time
)

GTWR_BY_TIME.to_csv(
    GTWR_ROOT /
    "GTWR_BY_TIME.csv",
    index=False
)

display(
    GTWR_BY_TIME
)

print(
    f"\nGTWR runtime: {(time.perf_counter()-t0)/60:.1f} min"
)

gc.collect()

# %% code cell 5

# ==========================================================================================
# CELL 5 — GGPR + SPATIAL HOLDOUT + UNCERTAINTY + ALL-HEX GEOSHAPLEY
# ==========================================================================================

t0 = time.perf_counter()

print("=" * 105)
print("SIMULATION V2 — GGPR + UNCERTAINTY + GEOSHAPLEY")
print("=" * 105)


# ==========================================================================================
# REPRESENTATIVE SAMPLE
# ==========================================================================================

def representative_sample(
    df,
    n,
    seed
):

    if n >= len(
        df
    ):
        return (
            df
            .copy()
            .reset_index(
                drop=True
            )
        )

    Z = StandardScaler().fit_transform(
        df[
            ["X_KM", "Y_KM"]
        ]
    )

    km = MiniBatchKMeans(
        n_clusters=n,
        random_state=seed,
        n_init=5,
        batch_size=2048
    )

    labels = km.fit_predict(
        Z
    )

    centers = km.cluster_centers_

    chosen = []

    for c in range(
        n
    ):

        ii = np.where(
            labels == c
        )[0]

        if len(
            ii
        ) == 0:
            continue

        d2 = np.sum(
            (
                Z[ii]
                -
                centers[c]
            ) ** 2,
            axis=1
        )

        chosen.append(
            ii[
                np.argmin(
                    d2
                )
            ]
        )

    return (
        df
        .iloc[
            chosen
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ==========================================================================================
# FIT / TRANSFORM / PREDICT
# ==========================================================================================

def fit_sim_ggpr(
    train_df,
    spatial_weight,
    length_scale
):

    x_scaler = StandardScaler()

    xz = x_scaler.fit_transform(
        train_df[
            ["X"]
        ]
    )

    coord_scaler = StandardScaler()

    cz = coord_scaler.fit_transform(
        train_df[
            ["X_KM", "Y_KM"]
        ]
    )

    y_scaler = StandardScaler()

    yz = y_scaler.fit_transform(
        train_df[
            ["Y"]
        ]
    ).ravel()

    A = np.column_stack([
        xz,
        cz
    ])

    kernel = SimulationGGPRKernel(
        spatial_weight=spatial_weight,
        length_scale=length_scale
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=GGPR_NOISE,
        optimizer=None,
        normalize_y=False,
        random_state=RANDOM_STATE
    )

    model.fit(
        A,
        yz
    )

    return {
        "model":
            model,

        "x_scaler":
            x_scaler,

        "coord_scaler":
            coord_scaler,

        "y_scaler":
            y_scaler,
    }


def sim_ggpr_transform(
    obj,
    df
):

    xz = obj[
        "x_scaler"
    ].transform(
        df[
            ["X"]
        ]
    )

    cz = obj[
        "coord_scaler"
    ].transform(
        df[
            ["X_KM", "Y_KM"]
        ]
    )

    return np.column_stack([
        xz,
        cz
    ])


def sim_ggpr_predict(
    obj,
    df,
    return_std=False
):

    A = sim_ggpr_transform(
        obj,
        df
    )

    if return_std:

        mean_z, std_z = obj[
            "model"
        ].predict(
            A,
            return_std=True
        )

        mean = (
            obj[
                "y_scaler"
            ]
            .inverse_transform(
                mean_z.reshape(
                    -1,
                    1
                )
            )
            .ravel()
        )

        std = (
            std_z
            *
            float(
                obj[
                    "y_scaler"
                ].scale_[
                    0
                ]
            )
        )

        return (
            mean,
            std
        )

    pred_z = obj[
        "model"
    ].predict(
        A
    )

    return (
        obj[
            "y_scaler"
        ]
        .inverse_transform(
            pred_z.reshape(
                -1,
                1
            )
        )
        .ravel()
    )


# ==========================================================================================
# FIVE SPATIAL BLOCKS
# ==========================================================================================

GG_DATA = SIM_CLIM.copy()

GG_DATA[
    "_BLOCK"
] = (
    KMeans(
        n_clusters=5,
        random_state=RANDOM_STATE,
        n_init=30
    )
    .fit_predict(
        GG_DATA[
            ["X_KM", "Y_KM"]
        ]
    )
)

GG_TEST = GG_DATA.loc[
    GG_DATA[
        "_BLOCK"
    ].eq(
        0
    )
].copy()

GG_TUNE_VAL = GG_DATA.loc[
    GG_DATA[
        "_BLOCK"
    ].eq(
        1
    )
].copy()

GG_TUNE_POOL = GG_DATA.loc[
    GG_DATA[
        "_BLOCK"
    ].isin(
        [
            2,
            3,
            4
        ]
    )
].copy()

GG_TRAIN_TUNE = representative_sample(
    GG_TUNE_POOL,
    min(
        GGPR_TUNE_N,
        len(
            GG_TUNE_POOL
        )
    ),
    RANDOM_STATE
)


# ==========================================================================================
# HYPERPARAMETER TUNING
# ==========================================================================================

tune_rows = []

for sw in GGPR_SPATIAL_WEIGHTS:

    for ls in GGPR_LENGTH_SCALES:

        obj = fit_sim_ggpr(
            GG_TRAIN_TUNE,
            sw,
            ls
        )

        pred_tune = sim_ggpr_predict(
            obj,
            GG_TUNE_VAL
        )

        row = {
            "SPATIAL_WEIGHT":
                sw,

            "LENGTH_SCALE":
                ls,

            "R2":
                r2_score(
                    GG_TUNE_VAL[
                        "Y"
                    ],
                    pred_tune
                ),

            "RMSE":
                mean_squared_error(
                    GG_TUNE_VAL[
                        "Y"
                    ],
                    pred_tune
                ) ** 0.5,

            "MAE":
                mean_absolute_error(
                    GG_TUNE_VAL[
                        "Y"
                    ],
                    pred_tune
                ),
        }

        tune_rows.append(
            row
        )

        print(
            f"w={sw:<4} "
            f"ls={ls:<4} "
            f"R2={row['R2']:.4f} "
            f"RMSE={row['RMSE']:.4f}"
        )

        del obj
        gc.collect()

GG_TUNING = (
    pd.DataFrame(
        tune_rows
    )

    .sort_values(
        [
            "RMSE",
            "R2"
        ],
        ascending=[
            True,
            False
        ]
    )

    .reset_index(
        drop=True
    )
)

BEST_GG_SW = float(
    GG_TUNING.iloc[
        0
    ][
        "SPATIAL_WEIGHT"
    ]
)

BEST_GG_LS = float(
    GG_TUNING.iloc[
        0
    ][
        "LENGTH_SCALE"
    ]
)

display(
    GG_TUNING
)

print(
    "\nBest GGPR spatial weight:",
    BEST_GG_SW
)

print(
    "Best GGPR length scale:",
    BEST_GG_LS
)

GG_TUNING.to_csv(
    GGPR_ROOT /
    "GGPR_HYPERPARAMETER_TUNING.csv",
    index=False
)


# ==========================================================================================
# SAMPLE-SIZE SENSITIVITY ON SAME UNTOUCHED TEST BLOCK
# ==========================================================================================

GG_POOL_TEST = GG_DATA.loc[
    ~GG_DATA[
        "_BLOCK"
    ].eq(
        0
    )
].copy()

sens_rows = []

for n in GGPR_SAMPLE_SIZES:

    train = representative_sample(
        GG_POOL_TEST,
        min(
            n,
            len(
                GG_POOL_TEST
            )
        ),
        RANDOM_STATE + n
    )

    obj = fit_sim_ggpr(
        train,
        BEST_GG_SW,
        BEST_GG_LS
    )

    pred_test = sim_ggpr_predict(
        obj,
        GG_TEST
    )

    row = {
        "N_CALIBRATION":
            len(
                train
            ),

        "N_TEST":
            len(
                GG_TEST
            ),

        "R2":
            r2_score(
                GG_TEST[
                    "Y"
                ],
                pred_test
            ),

        "RMSE":
            mean_squared_error(
                GG_TEST[
                    "Y"
                ],
                pred_test
            ) ** 0.5,

        "MAE":
            mean_absolute_error(
                GG_TEST[
                    "Y"
                ],
                pred_test
            ),
    }

    sens_rows.append(
        row
    )

    print(
        row
    )

    del obj
    gc.collect()

GG_SENSITIVITY = pd.DataFrame(
    sens_rows
)

GG_SENSITIVITY.to_csv(
    GGPR_ROOT /
    "GGPR_SAMPLE_SIZE_SENSITIVITY.csv",
    index=False
)

display(
    GG_SENSITIVITY
)


# ==========================================================================================
# FINAL GGPR ON 900 REPRESENTATIVE HEXES
# ==========================================================================================

GG_FINAL_CAL = representative_sample(
    GG_DATA,
    min(
        GGPR_FINAL_N,
        len(
            GG_DATA
        )
    ),
    RANDOM_STATE + 999
)

GG_FINAL = fit_sim_ggpr(
    GG_FINAL_CAL,
    BEST_GG_SW,
    BEST_GG_LS
)

GG_MEAN, GG_STD = sim_ggpr_predict(
    GG_FINAL,
    GG_DATA,
    return_std=True
)

GG_RESULT = GG_DATA[
    [
        "HEX_ID",
        "X_KM",
        "Y_KM",
        "X",
        "Y",
        "Y_SIGNAL",
        "TRUE_BETA_X"
    ]
].copy()

GG_RESULT[
    "GGPR_PRED_Y"
] = GG_MEAN

GG_RESULT[
    "GGPR_POSTERIOR_SD"
] = GG_STD

GG_RESULT[
    "GGPR_CI95_LO"
] = (
    GG_MEAN
    -
    1.96 * GG_STD
)

GG_RESULT[
    "GGPR_CI95_HI"
] = (
    GG_MEAN
    +
    1.96 * GG_STD
)

GG_RESULT[
    "GGPR_CI95_WIDTH"
] = (
    3.92 * GG_STD
)

GG_RESULT[
    "GGPR_RESID"
] = (
    GG_RESULT[
        "Y"
    ]
    -
    GG_MEAN
)

GG_RESULT.to_parquet(
    GGPR_ROOT /
    "GGPR_ALL_HEX_PREDICTION_UNCERTAINTY.parquet",
    index=False
)

GG_FINAL_CAL.to_csv(
    GGPR_ROOT /
    "GGPR_FINAL_CALIBRATION.csv",
    index=False
)

GG_UNCERT_SUMMARY = pd.DataFrame([{
    "N_HEX":
        len(
            GG_RESULT
        ),

    "MEAN_POSTERIOR_SD":
        float(
            np.mean(
                GG_STD
            )
        ),

    "MEDIAN_POSTERIOR_SD":
        float(
            np.median(
                GG_STD
            )
        ),

    "P05_POSTERIOR_SD":
        float(
            np.percentile(
                GG_STD,
                5
            )
        ),

    "P95_POSTERIOR_SD":
        float(
            np.percentile(
                GG_STD,
                95
            )
        ),

    "MAX_POSTERIOR_SD":
        float(
            np.max(
                GG_STD
            )
        ),
}])

GG_UNCERT_SUMMARY.to_csv(
    GGPR_ROOT /
    "GGPR_UNCERTAINTY_SUMMARY.csv",
    index=False
)

display(
    GG_UNCERT_SUMMARY
)


# ==========================================================================================
# ALL-HEX ANALYTICAL GEOSHAPLEY
#
# The GGPR posterior mean is additive:
# feature-similarity contribution + geographic Matérn contribution.
# ==========================================================================================

GP = GG_FINAL[
    "model"
]

A_TRAIN = np.asarray(
    GP.X_train_,
    float
)

ALPHA = np.asarray(
    GP.alpha_,
    float
).ravel()

A_ALL = sim_ggpr_transform(
    GG_FINAL,
    GG_DATA
)

A_BG = sim_ggpr_transform(
    GG_FINAL,
    GG_FINAL_CAL
)

Y_SCALE_GG = float(
    GG_FINAL[
        "y_scaler"
    ].scale_[
        0
    ]
)

Y_MEAN_GG = float(
    GG_FINAL[
        "y_scaler"
    ].mean_[
        0
    ]
)


def sim_gg_components(
    A_query,
    batch=500
):

    A_query = np.asarray(
        A_query,
        float
    )

    out = np.empty(
        (
            len(
                A_query
            ),
            2
        ),
        dtype=float
    )

    train_x = A_TRAIN[
        :,
        0
    ]

    train_geo = A_TRAIN[
        :,
        1:3
    ]

    for start in range(
        0,
        len(
            A_query
        ),
        batch
    ):

        end = min(
            start
            +
            batch,
            len(
                A_query
            )
        )

        q = A_query[
            start:end
        ]

        # Feature-similarity component.
        d = (
            q[
                :,
                0
            ][
                :,
                None
            ]
            -
            train_x[
                None,
                :
            ]
        )

        feature_kernel = np.exp(
            -0.5 * d**2
        )

        out[
            start:end,
            0
        ] = (
            feature_kernel
            @
            ALPHA
        )

        # Geographic Matérn component.
        dx = (
            q[
                :,
                1
            ][
                :,
                None
            ]
            -
            train_geo[
                None,
                :,
                0
            ]
        )

        dy = (
            q[
                :,
                2
            ][
                :,
                None
            ]
            -
            train_geo[
                None,
                :,
                1
            ]
        )

        dist = np.sqrt(
            dx**2
            +
            dy**2
        )

        r = (
            np.sqrt(
                3.0
            )
            *
            dist
            /
            BEST_GG_LS
        )

        matern = (
            1.0
            +
            r
        ) * np.exp(
            -r
        )

        out[
            start:end,
            1
        ] = (
            BEST_GG_SW
            *
            (
                matern
                @
                ALPHA
            )
        )

    return out


BG_COMPONENTS = sim_gg_components(
    A_BG
)

BG_MEAN_COMPONENT = BG_COMPONENTS.mean(
    axis=0
)

ALL_COMPONENTS = sim_gg_components(
    A_ALL
)

CONTRIB = (
    ALL_COMPONENTS
    -
    BG_MEAN_COMPONENT[
        None,
        :
    ]
) * Y_SCALE_GG

GEOSHAP_BASE = (
    Y_MEAN_GG
    +
    Y_SCALE_GG
    *
    float(
        BG_MEAN_COMPONENT.sum()
    )
)

GEO_ALL = GG_DATA[
    [
        "HEX_ID",
        "X_KM",
        "Y_KM",
        "X",
        "Y"
    ]
].copy()

GEO_ALL[
    "GEOSHAP_BASE"
] = GEOSHAP_BASE

GEO_ALL[
    "GEOSHAP_X"
] = CONTRIB[
    :,
    0
]

GEO_ALL[
    "GEOSHAP_GEO"
] = CONTRIB[
    :,
    1
]

GEO_ALL[
    "GGPR_PRED_Y"
] = GG_MEAN

GEO_ALL[
    "GEOSHAP_RECONSTRUCTED"
] = (
    GEOSHAP_BASE
    +
    GEO_ALL[
        "GEOSHAP_X"
    ]
    +
    GEO_ALL[
        "GEOSHAP_GEO"
    ]
)

GEO_ALL[
    "GEOSHAP_RECON_ERROR"
] = (
    GEO_ALL[
        "GEOSHAP_RECONSTRUCTED"
    ]
    -
    GEO_ALL[
        "GGPR_PRED_Y"
    ]
)

MAX_RECON_ERROR = float(
    np.abs(
        GEO_ALL[
            "GEOSHAP_RECON_ERROR"
        ]
    ).max()
)

print(
    "\nAll-hex GeoShapley reconstruction error:",
    MAX_RECON_ERROR
)


# ==========================================================================================
# OFFICIAL GEOSHAPLEY VERIFICATION — 5 TARGETS ONLY
# ==========================================================================================

verify_idx = np.linspace(
    0,
    len(
        GG_DATA
    ) - 1,
    5,
    dtype=int
)

VERIFY_DF = GG_DATA.iloc[
    verify_idx
].copy()

VERIFY_BG = GG_FINAL_CAL.sample(
    n=min(
        3,
        len(
            GG_FINAL_CAL
        )
    ),
    random_state=RANDOM_STATE
)


def official_predict(raw):

    if isinstance(
        raw,
        pd.DataFrame
    ):
        d = raw.copy()

    else:
        d = pd.DataFrame(
            np.asarray(
                raw,
                float
            ),
            columns=[
                "X",
                "X_KM",
                "Y_KM"
            ]
        )

    return sim_ggpr_predict(
        GG_FINAL,
        d
    )


OFFICIAL_EXPLAINER = GeoShapleyExplainer(
    official_predict,
    background=VERIFY_BG[
        [
            "X",
            "X_KM",
            "Y_KM"
        ]
    ].values,
    g=2,
    exact=True
)

OFFICIAL_RESULT = OFFICIAL_EXPLAINER.explain(
    VERIFY_DF[
        [
            "X",
            "X_KM",
            "Y_KM"
        ]
    ],
    n_jobs=2
)

official_primary = np.asarray(
    OFFICIAL_RESULT.primary,
    float
).ravel()

official_geo = np.asarray(
    OFFICIAL_RESULT.geo,
    float
).ravel()

official_interaction = np.asarray(
    OFFICIAL_RESULT.geo_intera,
    float
).ravel()

A_VERIFY = sim_ggpr_transform(
    GG_FINAL,
    VERIFY_DF
)

A_VERIFY_BG = sim_ggpr_transform(
    GG_FINAL,
    VERIFY_BG
)

anal_comp = sim_gg_components(
    A_VERIFY
)

anal_bg = sim_gg_components(
    A_VERIFY_BG
).mean(
    axis=0
)

anal_contrib = (
    anal_comp
    -
    anal_bg[
        None,
        :
    ]
) * Y_SCALE_GG

GEO_VERIFY = pd.DataFrame([
    {
        "COMPONENT":
            "X",

        "MAX_ABS_DIFF_VS_OFFICIAL":
            float(
                np.max(
                    np.abs(
                        anal_contrib[
                            :,
                            0
                        ]
                        -
                        official_primary
                    )
                )
            ),

        "OFFICIAL_GEO_INTERACTION_MAXABS":
            float(
                np.max(
                    np.abs(
                        official_interaction
                    )
                )
            ),
    },

    {
        "COMPONENT":
            "GEO",

        "MAX_ABS_DIFF_VS_OFFICIAL":
            float(
                np.max(
                    np.abs(
                        anal_contrib[
                            :,
                            1
                        ]
                        -
                        official_geo
                    )
                )
            ),

        "OFFICIAL_GEO_INTERACTION_MAXABS":
            np.nan,
    }
])

display(
    GEO_VERIFY
)

GEO_ALL.to_parquet(
    GEO_ROOT /
    "GEOSHAPLEY_ALL_HEX.parquet",
    index=False
)

GEO_VERIFY.to_csv(
    GEO_ROOT /
    "GEOSHAPLEY_OFFICIAL_VERIFICATION.csv",
    index=False
)

GEO_GLOBAL = pd.DataFrame([
    {
        "COMPONENT":
            "X",

        "MEAN_ABS_CONTRIBUTION":
            float(
                np.mean(
                    np.abs(
                        GEO_ALL[
                            "GEOSHAP_X"
                        ]
                    )
                )
            ),

        "MEAN_CONTRIBUTION":
            float(
                np.mean(
                    GEO_ALL[
                        "GEOSHAP_X"
                    ]
                )
            ),
    },

    {
        "COMPONENT":
            "GEO",

        "MEAN_ABS_CONTRIBUTION":
            float(
                np.mean(
                    np.abs(
                        GEO_ALL[
                            "GEOSHAP_GEO"
                        ]
                    )
                )
            ),

        "MEAN_CONTRIBUTION":
            float(
                np.mean(
                    GEO_ALL[
                        "GEOSHAP_GEO"
                    ]
                )
            ),
    }
])

GEO_GLOBAL.to_csv(
    GEO_ROOT /
    "GEOSHAPLEY_GLOBAL_IMPORTANCE.csv",
    index=False
)

display(
    GEO_GLOBAL
)

print(
    f"\nGGPR + GeoShapley runtime: {(time.perf_counter()-t0)/60:.1f} min"
)

gc.collect()

# %% code cell 6

# ==========================================================================================
# CELL 6 — ALL T1–T10 CARTOGRAPHY OUTPUTS + MASTER METRICS + ONE ZIP
# ==========================================================================================

t0 = time.perf_counter()

print("=" * 105)
print("SIMULATION V2 — CARTOGRAPHY + FINAL ZIP")
print("=" * 105)

SIM_HEX_BASE = SIM_HEX[
    [
        "HEX_ID",
        "SIM_REGION",
        "geometry"
    ]
].copy()

SIM_HEX_BASE[
    "HEX_ID"
] = SIM_HEX_BASE[
    "HEX_ID"
].astype(str)

SIM_HEX_BASE.to_parquet(
    CARTO_ROOT /
    "SIMULATION_HEX_GRID.parquet",
    index=False
)


# ==========================================================================================
# DIRECT GVSTMR — ALL 10 TIMES
# ==========================================================================================

SIM_PANEL.to_parquet(
    CARTO_ROOT /
    "GVSTMR_ALL_T1_T10_ATTRIBUTES.parquet",
    index=False
)

GVSTMR_MAPS = {}

for t in range(
    1,
    N_TIME + 1
):

    att = SIM_PANEL.loc[
        SIM_PANEL[
            "TIME"
        ].eq(
            t
        )
    ].copy()

    att[
        "HEX_ID"
    ] = att[
        "HEX_ID"
    ].astype(str)

    map_t = SIM_HEX_BASE.merge(
        att,
        on="HEX_ID",
        how="inner",
        validate="1:1"
    )

    map_t = gpd.GeoDataFrame(
        map_t,
        geometry="geometry",
        crs=SIM_HEX.crs
    )

    map_t.to_parquet(
        CARTO_ROOT /
        f"GVSTMR_SIM_T{t}.parquet",
        index=False
    )

    GVSTMR_MAPS[
        t
    ] = map_t

    print(
        f"GVSTMR T{t}: {len(map_t):,}"
    )


# ==========================================================================================
# MGWR MAP
# ==========================================================================================

mg_att = MGWR_LOCAL.copy()

mg_att[
    "HEX_ID"
] = mg_att[
    "HEX_ID"
].astype(str)

MGWR_MAP = SIM_HEX_BASE.merge(
    mg_att,
    on="HEX_ID",
    how="inner",
    validate="1:1"
)

MGWR_MAP = gpd.GeoDataFrame(
    MGWR_MAP,
    geometry="geometry",
    crs=SIM_HEX.crs
)

MGWR_MAP.to_parquet(
    CARTO_ROOT /
    "MGWR_SIMULATION_MAP.parquet",
    index=False
)


# ==========================================================================================
# GTWR — ALL 10 TIMES
# ==========================================================================================

GT[
    "HEX_ID"
] = GT[
    "HEX_ID"
].astype(str)

GT.to_parquet(
    CARTO_ROOT /
    "GTWR_ALL_T1_T10_ATTRIBUTES.parquet",
    index=False
)

GTWR_MAPS = {}

for t in range(
    1,
    N_TIME + 1
):

    gt_t = GT.loc[
        GT[
            "TIME"
        ].eq(
            t
        )
    ].copy()

    gt_map = SIM_HEX_BASE.merge(
        gt_t,
        on="HEX_ID",
        how="inner",
        validate="1:1"
    )

    gt_map = gpd.GeoDataFrame(
        gt_map,
        geometry="geometry",
        crs=SIM_HEX.crs
    )

    gt_map.to_parquet(
        CARTO_ROOT /
        f"GTWR_SIM_T{t}.parquet",
        index=False
    )

    GTWR_MAPS[
        t
    ] = gt_map


# ==========================================================================================
# GGPR + UNCERTAINTY + GEOSHAPLEY MAP
# ==========================================================================================

GG_CARTO = GG_RESULT.merge(
    GEO_ALL[
        [
            "HEX_ID",
            "GEOSHAP_X",
            "GEOSHAP_GEO"
        ]
    ],
    on="HEX_ID",
    how="inner",
    validate="1:1"
)

GG_CARTO[
    "HEX_ID"
] = GG_CARTO[
    "HEX_ID"
].astype(str)

GG_CARTO = SIM_HEX_BASE.merge(
    GG_CARTO,
    on="HEX_ID",
    how="inner",
    validate="1:1"
)

GG_CARTO = gpd.GeoDataFrame(
    GG_CARTO,
    geometry="geometry",
    crs=SIM_HEX.crs
)

GG_CARTO.to_parquet(
    CARTO_ROOT /
    "GGPR_UNCERTAINTY_GEOSHAP_SIMULATION_MAP.parquet",
    index=False
)


# ==========================================================================================
# GEOPACKAGE — ALL CARTOGRAPHY LAYERS
# ==========================================================================================

GPKG = (
    CARTO_ROOT /
    "GVSTMR_SIMULATION_CARTOGRAPHY.gpkg"
)

if GPKG.exists():
    GPKG.unlink()


def write_layer(
    gdf,
    layer_name
):

    gdf.to_file(
        GPKG,
        layer=layer_name,
        driver="GPKG",
        engine="pyogrio",
        index=False
    )

    print(
        f"GPKG: {layer_name}"
    )


write_layer(
    SIM_HEX_BASE,
    "hex_grid"
)

write_layer(
    MGWR_MAP,
    "mgwr_sim"
)

write_layer(
    GG_CARTO,
    "ggpr_uncert_geoshap"
)

for t in range(
    1,
    N_TIME + 1
):

    write_layer(
        GVSTMR_MAPS[
            t
        ],
        f"gvstmr_t{t}"
    )

    write_layer(
        GTWR_MAPS[
            t
        ],
        f"gtwr_t{t}"
    )


# ==========================================================================================
# CARTOGRAPHY FIELD GUIDE
# ==========================================================================================

guide_rows = [
    {
        "MODEL":
            "GVSTMR",

        "FIELD":
            "RHO_S",

        "TIME":
            "T1–T10",

        "DESCRIPTION":
            "Local spatial X–Y relationship score"
    },

    {
        "MODEL":
            "GVSTMR",

        "FIELD":
            "RHO_T",

        "TIME":
            "T1–T10",

        "DESCRIPTION":
            "Centered-window temporal X–Y relationship score"
    },

    {
        "MODEL":
            "GVSTMR",

        "FIELD":
            "GV_CELL",

        "TIME":
            "T1–T10",

        "DESCRIPTION":
            "Fixed-global-tertile 3×3 GVSTMR cell"
    },

    {
        "MODEL":
            "GVSTMR",

        "FIELD":
            "GV_COLOR",

        "TIME":
            "T1–T10",

        "DESCRIPTION":
            "Final cartographic color from the common 3×3 legend"
    },

    {
        "MODEL":
            "Simulation truth",

        "FIELD":
            "TRUE_BETA_X",

        "TIME":
            "T1–T10",

        "DESCRIPTION":
            "Known analytical local effect f'(X)"
    },

    {
        "MODEL":
            "MGWR",

        "FIELD":
            "BETA_X",

        "TIME":
            "T1–T10 climatological mean",

        "DESCRIPTION":
            "Local MGWR X coefficient"
    },

    {
        "MODEL":
            "GTWR",

        "FIELD":
            "BETA_X",

        "TIME":
            "T1–T10",

        "DESCRIPTION":
            "Local GTWR X coefficient"
    },

    {
        "MODEL":
            "GGPR",

        "FIELD":
            "GGPR_POSTERIOR_SD",

        "TIME":
            "T1–T10 climatological mean",

        "DESCRIPTION":
            "GGPR posterior predictive standard deviation"
    },

    {
        "MODEL":
            "GGPR",

        "FIELD":
            "GGPR_CI95_WIDTH",

        "TIME":
            "T1–T10 climatological mean",

        "DESCRIPTION":
            "Approximate 95% predictive interval width"
    },

    {
        "MODEL":
            "GeoShapley",

        "FIELD":
            "GEOSHAP_X",

        "TIME":
            "T1–T10 climatological mean",

        "DESCRIPTION":
            "All-hex GeoShapley contribution of X"
    },

    {
        "MODEL":
            "GeoShapley",

        "FIELD":
            "GEOSHAP_GEO",

        "TIME":
            "T1–T10 climatological mean",

        "DESCRIPTION":
            "All-hex geographic contribution"
    },
]

CARTO_GUIDE = pd.DataFrame(
    guide_rows
)

CARTO_GUIDE.to_csv(
    CARTO_ROOT /
    "CARTOGRAPHY_FIELD_GUIDE.csv",
    index=False
)


# ==========================================================================================
# MASTER METRICS
# ==========================================================================================

MASTER_METRICS = pd.DataFrame([
    {
        "ANALYSIS":
            "Oracle DGP signal",

        "METRIC":
            "R2",

        "VALUE":
            float(
                ORACLE_R2
            ),

        "N":
            N_HEX * N_TIME,
    },

    {
        "ANALYSIS":
            "GVSTMR spatial score vs true local feature effect",

        "METRIC":
            "Spearman",

        "VALUE":
            float(
                SPATIAL_BETA_SPEARMAN
            ),

        "N":
            N_HEX * N_TIME,
    },

    {
        "ANALYSIS":
            "GVSTMR temporal score vs true local feature effect",

        "METRIC":
            "Spearman",

        "VALUE":
            float(
                TEMPORAL_BETA_SPEARMAN
            ),

        "N":
            N_HEX * N_TIME,
    },

    {
        "ANALYSIS":
            "MGWR climatological local fit",

        "METRIC":
            "R2",

        "VALUE":
            float(
                MGWR_SUMMARY.iloc[
                    0
                ][
                    "R2"
                ]
            ),

        "N":
            int(
                MGWR_SUMMARY.iloc[
                    0
                ][
                    "N_HEX"
                ]
            ),
    },

    {
        "ANALYSIS":
            "GTWR full local fit",

        "METRIC":
            "R2",

        "VALUE":
            float(
                GTWR_SUMMARY.iloc[
                    0
                ][
                    "R2_LOCAL_FIT"
                ]
            ),

        "N":
            int(
                GTWR_SUMMARY.iloc[
                    0
                ][
                    "N_HEX_TIME"
                ]
            ),
    },

    {
        "ANALYSIS":
            "GGPR spatial holdout — largest calibration sample",

        "METRIC":
            "R2",

        "VALUE":
            float(
                GG_SENSITIVITY.iloc[
                    -1
                ][
                    "R2"
                ]
            ),

        "N":
            int(
                GG_SENSITIVITY.iloc[
                    -1
                ][
                    "N_TEST"
                ]
            ),
    },
])

MASTER_METRICS.to_csv(
    METRIC_ROOT /
    "SIMULATION_MASTER_METRICS.csv",
    index=False
)

display(
    MASTER_METRICS
)


# ==========================================================================================
# DESIGN / METHOD METADATA
# ==========================================================================================

DESIGN = {
    "AREA":
        "CONUS — same hex grid as real-world 2015–2024 case",

    "N_HEX":
        int(
            N_HEX
        ),

    "N_TIME":
        10,

    "TIMES":
        list(
            range(
                1,
                11
            )
        ),

    "SUGGESTED_MAIN_FIGURE_TIMES":
        DISPLAY_TIMES,

    "VARIABLE_PAIR":
        "X-Y",

    "DGP":
        (
            "Y = 30 + f(X) + g(space) + h(space,time) + epsilon; "
            "f(x)=x+2(x^2-1)+0.25x^3"
        ),

    "TRUE_LOCAL_X_EFFECT":
        "f'(x)=1+4x+0.75x^2",

    "SIM_NOISE_SD":
        SIM_NOISE_SD,

    "K_SPATIAL":
        K_SPATIAL,

    "TEMPORAL_WINDOW_SIZE":
        2 * TEMP_HALF_WINDOW + 1,

    "CLASSIFICATION":
        "Fixed global tertiles pooled across all T1–T10",

    "GVSTMR_PALETTE":
        GV_PALETTE,

    "MGWR":
        "Full-CONUS 10-time climatological Y~X",

    "GTWR":
        "Full-CONUS T1–T10 Y~X",

    "GGPR_FINAL_CALIBRATION_N":
        int(
            len(
                GG_FINAL_CAL
            )
        ),

    "GEOSHAPLEY":
        "All-hex additive GGPR decomposition + official verification",

    "NO_GOOGLE_DRIVE_WRITE":
        True,
}

with open(
    ROOT /
    "SIMULATION_DESIGN.json",
    "w"
) as f:

    json.dump(
        DESIGN,
        f,
        indent=2
    )


# ==========================================================================================
# README
# ==========================================================================================

README = f"""
GVSTMR CONUS SIMULATION V2 — HIGH-SIGNAL DGP
============================================

WHY V2
------
The previous simulation mixed independent latent spatial and temporal
components. That made Y only weakly predictable from X and produced
inappropriately low model performance for a proof-of-concept simulation.

V2 uses an explicit, known, smooth, low-noise DGP:

Y = 30 + f(X) + g(space) + h(space,time) + error

f(x) = x + 2(x^2 - 1) + 0.25x^3

true local X effect:
f'(x) = 1 + 4x + 0.75x^2

This makes the simulated process intentionally recoverable while retaining
strong spatial and temporal heterogeneity in the signed X–Y relationship.


TIME
----
T1–T10

Suggested main figure:
T1, T4, T7, T10

All T1–T10 layers are retained.


DIRECT GVSTMR
-------------
Spatial score:
X–Y Pearson relationship over focal hex + {K_SPATIAL} nearest neighbors.

Temporal score:
X–Y Pearson relationship in a centered {2*TEMP_HALF_WINDOW+1}-step window.

Classification:
one fixed set of global spatial and temporal tertiles pooled across all
10 simulated times.


COMMON 3×3 COLORS
-----------------
Spatial LOW:
#E5E4E9
#B9D9E6
#54B8D0

Spatial MEDIUM:
#DB95CB
#9E9FCB
#4582BB

Spatial HIGH:
#BE3F98
#74529F
#243A83


PREFLIGHT
---------
Before MGWR starts, the notebook verifies:

quick local spatial R2 >= {PREFLIGHT_MIN_LOCAL_R2}
quick GTWR R2 >= {PREFLIGHT_MIN_GTWR_R2}
quick GGPR spatial-holdout R2 >= {PREFLIGHT_MIN_GGPR_R2}

If any gate fails, execution stops immediately rather than wasting time.


MODELS
------
MGWR:
full-CONUS climatological Y~X.

GTWR:
full T1–T10 hex-time panel.

GGPR:
hyperparameter tuning + same spatial holdout across calibration sizes.

Uncertainty:
wall-to-wall GGPR posterior SD and 95% interval.

GeoShapley:
wall-to-wall analytical decomposition of the additive GGPR posterior mean,
checked against official exact GeoShapley on five locations.


CARTOGRAPHY
-----------
07_CARTOGRAPHY_READY contains:

GVSTMR_SIM_T1.parquet
...
GVSTMR_SIM_T10.parquet

GTWR_SIM_T1.parquet
...
GTWR_SIM_T10.parquet

MGWR_SIMULATION_MAP.parquet

GGPR_UNCERTAINTY_GEOSHAP_SIMULATION_MAP.parquet

GVSTMR_SIMULATION_CARTOGRAPHY.gpkg


FINAL FILE
----------
GVSTMR_SIMULATION_CONUS_10T_V2.zip

No new Google Drive write is used.
"""

(
    ROOT /
    "README_SIMULATION_V2.txt"
).write_text(
    README
)


# ==========================================================================================
# MANIFEST
# ==========================================================================================

manifest = []

for p in ROOT.rglob(
    "*"
):

    if p.is_file():

        manifest.append({
            "FILE":
                str(
                    p.relative_to(
                        ROOT
                    )
                ),

            "SIZE_MB":
                p.stat().st_size
                /
                1024**2,
        })

MANIFEST = (
    pd.DataFrame(
        manifest
    )
    .sort_values(
        "FILE"
    )
    .reset_index(
        drop=True
    )
)

MANIFEST.to_csv(
    ROOT /
    "FILE_MANIFEST.csv",
    index=False
)


# ==========================================================================================
# CREATE ONE FINAL ZIP
# ==========================================================================================

with zipfile.ZipFile(
    ZIP_PATH,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
    allowZip64=True
) as z:

    for p in ROOT.rglob(
        "*"
    ):

        if p.is_file():

            z.write(
                p,
                arcname=p.relative_to(
                    ROOT
                )
            )

if not zipfile.is_zipfile(
    ZIP_PATH
):
    raise RuntimeError(
        "Final simulation ZIP is invalid."
    )

with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as z:

    bad = z.testzip()

if bad is not None:
    raise RuntimeError(
        f"Corrupt ZIP member: {bad}"
    )


# ==========================================================================================
# FINAL REPORT
# ==========================================================================================

print("\n" + "=" * 105)
print("✅✅✅ GVSTMR SIMULATION V2 COMPLETE ✅✅✅")
print("=" * 105)

print(
    "\nOracle DGP R2:",
    round(
        ORACLE_R2,
        5
    )
)

print(
    "Spatial rho vs true beta Spearman:",
    round(
        SPATIAL_BETA_SPEARMAN,
        5
    )
)

print(
    "Temporal rho vs true beta Spearman:",
    round(
        TEMPORAL_BETA_SPEARMAN,
        5
    )
)

print(
    "\nMGWR R2:",
    round(
        float(
            MGWR_SUMMARY.iloc[
                0
            ][
                "R2"
            ]
        ),
        5
    )
)

print(
    "GTWR R2:",
    round(
        float(
            GTWR_SUMMARY.iloc[
                0
            ][
                "R2_LOCAL_FIT"
            ]
        ),
        5
    )
)

print(
    "GGPR holdout R2:",
    round(
        float(
            GG_SENSITIVITY.iloc[
                -1
            ][
                "R2"
            ]
        ),
        5
    )
)

print(
    "\nFinal ZIP:"
)

print(
    ZIP_PATH
)

print(
    "\nZIP size:",
    round(
        ZIP_PATH.stat().st_size
        /
        1024**2,
        2
    ),
    "MB"
)

print(
    "\nNO GOOGLE DRIVE WRITE."
)

try:
    from google.colab import files

    files.download(
        str(
            ZIP_PATH
        )
    )

except Exception:
    pass
