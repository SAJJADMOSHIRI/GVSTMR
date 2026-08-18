"""
Reproduce the core real-world GVSTMR case used in the revised manuscript.

Input
-----
GVSTMR_FINAL_2015_2024.zip

Core variable pair
------------------
NDVI <-> JJA land-surface temperature (LST_MEAN_C)

Period
------
2015-2024

Manuscript defaults
-------------------
Spatial neighborhood: focal hexagon + 30 nearest neighbors
Temporal kernel: Gaussian, bandwidth h=2
Classification: fixed pooled tertiles across all hexagons x all 10 years

Example
-------
python run_real_case.py --input-zip GVSTMR_FINAL_2015_2024.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gvstmr.core import build_direct_gvstmr, GV_PALETTE


YEARS = list(range(2015, 2025))
DISPLAY_YEARS = [2015, 2018, 2021, 2024]


def find_member(zip_path: Path, suffix: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        matches = [n for n in z.namelist() if n.endswith(suffix)]

    if not matches:
        raise FileNotFoundError(
            f"{suffix} not found inside {zip_path.name}"
        )

    return sorted(matches, key=lambda x: (x.count("/"), len(x)))[0]


def extract_member(zip_path: Path, member: str, target_root: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(member, target_root)
    return target_root / member


def draw_matrix_legend(ax):
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    for s in range(3):
        for t in range(3):
            cell = s * 3 + t + 1
            ax.add_patch(
                Rectangle(
                    (t, s),
                    1,
                    1,
                    facecolor=GV_PALETTE[cell],
                    edgecolor="white",
                    linewidth=1.0,
                )
            )

    ax.set_xticks([0.5, 1.5, 2.5], ["Low", "Medium", "High"])
    ax.set_yticks([0.5, 1.5, 2.5], ["Low", "Medium", "High"])
    ax.set_xlabel(r"Temporal relationship $\rho^T$")
    ax.set_ylabel(r"Spatial relationship $\rho^S$")
    ax.set_title("GVSTMR matrix")
    ax.tick_params(length=0)

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_aspect("equal")


def plot_snapshots(gdf: gpd.GeoDataFrame, output_png: Path):
    fig = plt.figure(figsize=(14, 4.6), facecolor="white")
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[1, 1, 1, 1, 0.72],
        left=0.02,
        right=0.98,
        top=0.86,
        bottom=0.08,
        wspace=0.04,
    )

    for i, year in enumerate(DISPLAY_YEARS):
        ax = fig.add_subplot(gs[0, i])
        q = gdf.loc[gdf["YEAR"].eq(year)]

        for cell in range(1, 10):
            part = q.loc[q["GV_CELL"].eq(cell)]
            if len(part):
                part.plot(
                    ax=ax,
                    facecolor=GV_PALETTE[cell],
                    edgecolor="none",
                )

        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(str(year), fontweight="bold")

    leg = fig.add_subplot(gs[0, 4])
    draw_matrix_legend(leg)

    fig.suptitle(
        "Real-world GVSTMR case: NDVI–JJA LST, CONUS 2015–2024",
        fontsize=12,
        fontweight="bold",
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-zip",
        default="GVSTMR_FINAL_2015_2024.zip",
        help="Path to GVSTMR_FINAL_2015_2024.zip",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/real_case",
        help="Output directory",
    )

    args = parser.parse_args()

    input_zip = Path(args.input_zip).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_zip.exists():
        raise FileNotFoundError(input_zip)

    work = output_dir / "_work"

    if work.exists():
        shutil.rmtree(work)

    work.mkdir(parents=True)

    panel_member = find_member(
        input_zip,
        "PANEL_2015_2024.parquet",
    )

    hex_member = find_member(
        input_zip,
        "HEX_GRID.parquet",
    )

    panel_path = extract_member(
        input_zip,
        panel_member,
        work,
    )

    hex_path = extract_member(
        input_zip,
        hex_member,
        work,
    )

    panel = pd.read_parquet(panel_path)
    hex_grid = gpd.read_parquet(hex_path)

    panel["HEX_ID"] = panel["HEX_ID"].astype(str)
    panel["YEAR"] = pd.to_numeric(
        panel["YEAR"],
        errors="coerce",
    ).astype("Int64")

    hex_grid["HEX_ID"] = hex_grid["HEX_ID"].astype(str)

    if hex_grid.crs is None:
        hex_grid = hex_grid.set_crs("EPSG:5070")
    elif hex_grid.crs.to_epsg() != 5070:
        hex_grid = hex_grid.to_crs(5070)

    required = {
        "HEX_ID",
        "YEAR",
        "NDVI",
        "LST_MEAN_C",
    }

    missing = sorted(required - set(panel.columns))

    if missing:
        raise RuntimeError(
            "Missing required real-case columns: "
            + ", ".join(missing)
        )

    direct, thresholds = build_direct_gvstmr(
        panel=panel,
        geometry=hex_grid,
        x_col="NDVI",
        y_col="LST_MEAN_C",
        time_col="YEAR",
        times=YEARS,
        k_spatial=30,
        temporal_bandwidth=2.0,
        extra_cols=[
            c
            for c in ["TEMP_C"]
            if c in panel.columns
        ],
    )

    out_parquet = (
        output_dir
        /
        "DIRECT_GVSTMR_REAL_NDVI_LST_2015_2024.parquet"
    )

    direct.to_parquet(
        out_parquet,
        index=False,
    )

    with open(
        output_dir / "REAL_GVSTMR_THRESHOLDS.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            thresholds,
            f,
            indent=2,
        )

    occupancy = (
        direct.groupby(
            ["YEAR", "GV_CELL"]
        )
        .size()
        .rename("N")
        .reset_index()
    )

    occupancy["SHARE"] = (
        occupancy["N"]
        /
        occupancy.groupby("YEAR")["N"].transform("sum")
    )

    occupancy.to_csv(
        output_dir / "REAL_GVSTMR_STATE_OCCUPANCY.csv",
        index=False,
    )

    plot_snapshots(
        direct,
        output_dir / "REAL_GVSTMR_SNAPSHOTS.png",
    )

    archive = shutil.make_archive(
        str(output_dir / "GVSTMR_REAL_CASE_OUTPUTS"),
        "zip",
        root_dir=output_dir,
    )

    shutil.rmtree(
        work,
        ignore_errors=True,
    )

    print("Complete.")
    print("Thresholds:", thresholds)
    print("Output:", out_parquet)
    print("Archive:", archive)


if __name__ == "__main__":
    main()
