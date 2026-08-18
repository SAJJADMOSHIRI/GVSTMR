# GVSTMR

**Geovisual Spatiotemporal Matrix Relationship (GVSTMR)**

This repository contains the code package for the revised GVSTMR manuscript.

## Repository structure

### `01_core_method/`
Minimal, reusable implementation of the manuscript's GVSTMR operator.

- `gvstmr_core.py`
- `GVSTMR_Core_Concept_Colab.ipynb`

The manuscript operator combines:

1. local spatial relationship \(\rho^S_{i,t}\);
2. time-local Gaussian-weighted temporal relationship \(\rho^T_{i,t}\);
3. fixed pooled tertile classification;
4. a common 3×3 cartographic state space.

### `02_real_case/`
Core real-world case:

**NDVI ↔ JJA land-surface temperature, CONUS, 2015–2024**

- `run_real_case.py`
- `GVSTMR_Real_Case_CONUS_2015_2024_Colab.ipynb`

Required input data archive:

`GVSTMR_FINAL_2015_2024.zip`

### `03_simulation/`
Full CONUS simulation T1–T10 used for known-process validation.

- `GVSTMR_Simulation_CONUS_T1_T10_Colab.ipynb`
- `GVSTMR_Simulation_CONUS_T1_T10.py`

The Colab notebook is the recommended executable version.

### `04_arcgis_toolbox/`
ArcGIS Pro Python Toolbox:

- `GVSTMR_ArcGISPro.pyt`

The current toolbox provides a practical **period-integrated one-map mode**.
See the folder README for the distinction between this operational mode and the
paper's time-resolved research operator.

### `05_paper_reproduction/`
Final notebooks used for manuscript cartography and Supplementary Material.

- `GVSTMR_Main_Figures_Reproduction.ipynb`
- `GVSTMR_Supplementary_Reproduction.ipynb`

These notebooks expect the analysis ZIP archives described below.

## Manuscript defaults

- Spatial neighborhood: focal hexagon + 30 nearest neighbors
- Temporal relationship: Gaussian-weighted Pearson correlation
- Temporal bandwidth: h = 2 time units
- Classification: fixed pooled tertiles within each case
- Matrix: 3 × 3
- Real-world variable pair: NDVI ↔ JJA LST
- Real-world period: 2015–2024
- Simulation period: T1–T10

## Data

Large input/output archives should **not** be committed to the normal Git repository.

The manuscript reproduction notebooks use:

1. `GVSTMR_FINAL_2015_2024.zip`
2. `GVSTMR_SIMULATION_CONUS_10T_V2.zip`
3. `GVSTMR_ANNUAL_GGPR_GEOSHAP_UNCERTAINTY_2015_2024.zip`

Archive these files separately (for example in the Zenodo record associated with the paper)
and place the DOI/link here after publication of the software/data release.

## Colab

Each `.ipynb` file is a standard Jupyter notebook and can be opened in Google Colab.

For the easiest route:

1. open the notebook in Colab;
2. upload the requested ZIP when prompted;
3. Run all.

## Citation

A `CITATION.cff` template is included. Add the full author list and the Zenodo DOI
before making the v1.0.0 manuscript release.

## License

Choose and add the final code license before the public release.
MIT or BSD-3-Clause are common permissive choices, but the final choice belongs to the authors.
