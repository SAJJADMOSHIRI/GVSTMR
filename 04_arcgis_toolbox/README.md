# ArcGIS Pro Toolbox

`GVSTMR_ArcGISPro.pyt` is the ArcGIS Pro implementation created for practical GIS use.

## Current operational behavior

The toolbox accepts multiple time-specific X/Y field pairs and produces a **single period-integrated GVSTMR map** with:

- `RHO_S`
- `RHO_T`
- `GVSTMR`

It supports optional normalization fields, spatial-neighbor K, quantile/equal-interval classification, and 3×3 / 5×5 / 9×9 grids.

It also contains two display choices:

- **Classified list**
- **Bivariate matrix** (uses an existing ArcGIS Pro Bivariate Colors layer as a renderer template when available)

## Important relationship to the manuscript

The manuscript's research notebooks use a **time-resolved GVSTMR operator** at each time `t`.
The ArcGIS toolbox is an operational **period-integrated summary mode** requested for practical GIS use.

Do not use the toolbox output as a substitute for reproducing the paper's time-resolved figures.
Use `01_core_method`, `02_real_case`, and `03_simulation` for the manuscript method itself.

## ArcGIS Pro

Add the `.pyt` file through the Catalog pane → Toolboxes → Add Toolbox.
