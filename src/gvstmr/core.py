"""
GVSTMR core operator used in the revised manuscript.

GVSTMR = Geovisual Spatiotemporal Matrix Relationship.

For a variable pair X-Y at location i and time t:

1. Spatial relationship rho^S_i,t
   Local Pearson correlation across the focal spatial unit and its K nearest
   neighboring units.

2. Temporal relationship rho^T_i,t
   Gaussian-weighted Pearson correlation across the full time series,
   centered on target time t with bandwidth h.

3. Fixed pooled classification
   Spatial and temporal scores are each classified by one set of pooled
   tertile thresholds computed across all spatial units x all times within
   a case. The thresholds do not change through time.

4. GVSTMR state
   GV_CELL = (SPATIAL_CLASS - 1) * 3 + TEMPORAL_CLASS
   producing states 1,...,9.

The implementation below mirrors the final manuscript/cartography notebook:
K=30 and h=2 are the manuscript defaults, but both are configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree


GV_PALETTE = {
    1: "#E5E4E9",
    2: "#B9D9E6",
    3: "#54B8D0",
    4: "#DB95CB",
    5: "#9E9FCB",
    6: "#4582BB",
    7: "#BE3F98",
    8: "#74529F",
    9: "#243A83",
}

GV_LABELS = {
    1: "Low spatial / Low temporal",
    2: "Low spatial / Medium temporal",
    3: "Low spatial / High temporal",
    4: "Medium spatial / Low temporal",
    5: "Medium spatial / Medium temporal",
    6: "Medium spatial / High temporal",
    7: "High spatial / Low temporal",
    8: "High spatial / Medium temporal",
    9: "High spatial / High temporal",
}


@dataclass(frozen=True)
class GVSTMRConfig:
    """Configuration for the manuscript GVSTMR operator."""
    k_spatial: int = 30
    temporal_bandwidth: float = 2.0
    spatial_crs: str = "EPSG:5070"


def ensure_xy_from_geometry(
    gdf: gpd.GeoDataFrame,
    spatial_crs: str = "EPSG:5070",
) -> gpd.GeoDataFrame:
    """
    Return a copy with X_KM and Y_KM centroid coordinates.

    The paper uses a projected CONUS coordinate system (EPSG:5070) before KNN.
    """
    g = gdf.copy()

    if g.crs is None:
        g = g.set_crs(spatial_crs)
    elif str(g.crs) != str(spatial_crs):
        try:
            if g.crs.to_epsg() != int(spatial_crs.split(":")[-1]):
                g = g.to_crs(spatial_crs)
        except Exception:
            g = g.to_crs(spatial_crs)

    if "X_KM" not in g.columns or "Y_KM" not in g.columns:
        cent = g.geometry.centroid
        g["X_KM"] = cent.x / 1000.0
        g["Y_KM"] = cent.y / 1000.0

    return g


def rowwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pearson correlation for corresponding rows of A and B.

    Each row is one local spatial neighborhood.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    A0 = A - A.mean(axis=1, keepdims=True)
    B0 = B - B.mean(axis=1, keepdims=True)

    num = np.sum(A0 * B0, axis=1)
    den = np.sqrt(
        np.sum(A0 ** 2, axis=1)
        *
        np.sum(B0 ** 2, axis=1)
    )

    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=den > 1e-12,
    )


def weighted_temporal_corr(
    X_mat: np.ndarray,
    Y_mat: np.ndarray,
    times: Optional[Sequence[float]] = None,
    bandwidth: float = 2.0,
) -> np.ndarray:
    """
    Time-local Gaussian-weighted Pearson correlation.

    Parameters
    ----------
    X_mat, Y_mat
        Arrays of shape [time, spatial_unit].
    times
        Numeric time positions. If omitted, 0,1,...,T-1 are used.
        For annual 2015-2024 data this is equivalent to using the year sequence
        up to a constant shift.
    bandwidth
        Gaussian temporal bandwidth h.

    Returns
    -------
    np.ndarray
        rho^T with shape [time, spatial_unit].
    """
    X_mat = np.asarray(X_mat, dtype=float)
    Y_mat = np.asarray(Y_mat, dtype=float)

    if X_mat.shape != Y_mat.shape:
        raise ValueError("X_mat and Y_mat must have identical shapes.")

    if times is None:
        t = np.arange(X_mat.shape[0], dtype=float)
    else:
        t = np.asarray(times, dtype=float)

    if len(t) != X_mat.shape[0]:
        raise ValueError("len(times) must equal the number of time rows.")

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    out = np.empty_like(X_mat, dtype=float)

    for c in range(len(t)):
        w = np.exp(
            -0.5
            *
            ((t - t[c]) / float(bandwidth)) ** 2
        )
        w = w / w.sum()

        mx = np.sum(w[:, None] * X_mat, axis=0)
        my = np.sum(w[:, None] * Y_mat, axis=0)

        dx = X_mat - mx[None, :]
        dy = Y_mat - my[None, :]

        cov = np.sum(w[:, None] * dx * dy, axis=0)
        vx = np.sum(w[:, None] * dx ** 2, axis=0)
        vy = np.sum(w[:, None] * dy ** 2, axis=0)

        den = np.sqrt(vx * vy)

        out[c] = np.divide(
            cov,
            den,
            out=np.zeros_like(cov),
            where=den > 1e-12,
        )

    return np.clip(out, -1.0, 1.0)


def classify3(
    values: np.ndarray,
    low_cut: float,
    high_cut: float,
) -> np.ndarray:
    """Classify signed relationship scores into Low / Medium / High."""
    return np.where(
        values <= low_cut,
        1,
        np.where(values <= high_cut, 2, 3),
    ).astype(np.int8)


def build_direct_gvstmr(
    panel: pd.DataFrame,
    geometry: gpd.GeoDataFrame,
    x_col: str,
    y_col: str,
    time_col: str,
    times: Sequence,
    k_spatial: int = 30,
    temporal_bandwidth: float = 2.0,
    extra_cols: Optional[Iterable[str]] = None,
    id_col: str = "HEX_ID",
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Compute the time-resolved manuscript GVSTMR operator.

    Notes
    -----
    * The local spatial neighborhood includes the focal unit plus K nearest
      other units, matching the final manuscript code.
    * Only units with complete X and Y observations for all requested times are
      retained, matching the paper analysis.
    * Tertile thresholds are pooled across all retained units and all requested
      times, separately for rho^S and rho^T.
    """
    if k_spatial < 1:
        raise ValueError("k_spatial must be >= 1.")

    extra_cols = list(extra_cols or [])

    P = panel.copy()
    P[id_col] = P[id_col].astype(str)

    G = ensure_xy_from_geometry(geometry).copy()
    G[id_col] = G[id_col].astype(str)

    static = (
        G[[id_col, "X_KM", "Y_KM", "geometry"]]
        .drop_duplicates(id_col)
        .copy()
    )

    P = P.loc[P[time_col].isin(times)].copy()

    complete = (
        P.groupby(id_col)
        .agg(
            N_TIME=(time_col, "nunique"),
            NX=(x_col, lambda s: int(s.notna().sum())),
            NY=(y_col, lambda s: int(s.notna().sum())),
        )
    )

    ids = complete.index[
        complete["N_TIME"].eq(len(times))
        &
        complete["NX"].eq(len(times))
        &
        complete["NY"].eq(len(times))
    ].astype(str).tolist()

    order = (
        static.loc[static[id_col].isin(ids), id_col]
        .astype(str)
        .tolist()
    )

    if len(order) < 3:
        raise ValueError("Too few complete spatial units for GVSTMR.")

    static = (
        static
        .set_index(id_col)
        .loc[order]
        .reset_index()
    )

    pp = P.loc[P[id_col].isin(order)].copy()

    X = (
        pp.pivot(index=time_col, columns=id_col, values=x_col)
        .loc[list(times), order]
        .to_numpy(float)
    )

    Y = (
        pp.pivot(index=time_col, columns=id_col, values=y_col)
        .loc[list(times), order]
        .to_numpy(float)
    )

    coords = static[["X_KM", "Y_KM"]].to_numpy(float)

    tree = cKDTree(coords)
    _, nbr = tree.query(
        coords,
        k=min(k_spatial + 1, len(coords)),
    )

    if nbr.ndim == 1:
        nbr = nbr[:, None]

    rho_s = np.empty_like(X, dtype=float)

    for ti in range(len(times)):
        rho_s[ti] = rowwise_corr(
            X[ti][nbr],
            Y[ti][nbr],
        )

    rho_t = weighted_temporal_corr(
        X,
        Y,
        times=np.arange(len(times), dtype=float),
        bandwidth=temporal_bandwidth,
    )

    s_low, s_high = np.quantile(
        rho_s.ravel(),
        [1 / 3, 2 / 3],
    )

    t_low, t_high = np.quantile(
        rho_t.ravel(),
        [1 / 3, 2 / 3],
    )

    s_class = classify3(rho_s, s_low, s_high)
    t_class = classify3(rho_t, t_low, t_high)

    cell = (
        (s_class - 1) * 3 + t_class
    ).astype(np.int8)

    rows = []

    for ti, tv in enumerate(times):
        D = pd.DataFrame({
            id_col: order,
            time_col: tv,
            x_col: X[ti],
            y_col: Y[ti],
            "RHO_S": rho_s[ti],
            "RHO_T": rho_t[ti],
            "SPATIAL_CLASS": s_class[ti],
            "TEMPORAL_CLASS": t_class[ti],
            "GV_CELL": cell[ti],
        })

        for col in extra_cols:
            if col in pp.columns:
                vals = (
                    pp.loc[pp[time_col].eq(tv)]
                    .set_index(id_col)
                    .reindex(order)[col]
                    .to_numpy()
                )
                D[col] = vals

        D = static.merge(
            D,
            on=id_col,
            how="inner",
            validate="1:1",
        )

        D = gpd.GeoDataFrame(
            D,
            geometry="geometry",
            crs=G.crs,
        )

        rows.append(D)

    OUT = pd.concat(
        rows,
        ignore_index=True,
    )

    OUT = gpd.GeoDataFrame(
        OUT,
        geometry="geometry",
        crs=G.crs,
    )

    thresholds = {
        "SPATIAL_LOW": float(s_low),
        "SPATIAL_HIGH": float(s_high),
        "TEMPORAL_LOW": float(t_low),
        "TEMPORAL_HIGH": float(t_high),
        "K_SPATIAL": int(k_spatial),
        "TEMPORAL_BANDWIDTH": float(temporal_bandwidth),
        "N_COMPLETE_UNITS": int(len(order)),
        "N_TIMES": int(len(times)),
    }

    return OUT, thresholds


def state_label(cell: int) -> str:
    """Return a human-readable label for a GVSTMR state 1...9."""
    return GV_LABELS[int(cell)]
