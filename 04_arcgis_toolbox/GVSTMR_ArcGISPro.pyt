# -*- coding: utf-8 -*-
"""
GVSTMR_ArcGISPro.pyt

ONE-output GVSTMR toolbox.

Concept:
    Multiple input times -> ONE period-integrated spatial relationship rho^S
                          -> ONE temporal relationship rho^T
                          -> ONE GVSTMR matrix class per polygon

No time-specific GV_T01/GV_T02/... fields are created.

Output fields:
    RHO_S
    RHO_T
    GVSTMR

Spatial integration:
    For every time t, calculate the local 2x2 covariance matrix of X and Y
    across the K nearest OTHER polygons around focal polygon i:

        Sigma^S_i,t = [[var(X), cov(X,Y)],
                       [cov(X,Y), var(Y)]]

    Then integrate through the full requested period by averaging the
    covariance-matrix elements over all valid times:

        Sigma^S_i = mean_t(Sigma^S_i,t)

    Finally:

        rho^S_i = Sigma^S_xy /
                  sqrt(Sigma^S_xx * Sigma^S_yy)

Temporal relationship:
    For each polygon i, use ALL requested times as the temporal sample:

        Sigma^T_i = covariance of
                    [(X_i,1,Y_i,1), ..., (X_i,T,Y_i,T)]

        rho^T_i = Sigma^T_xy /
                  sqrt(Sigma^T_xx * Sigma^T_yy)

The final GVSTMR field is created by classifying rho^S and rho^T once,
across all polygons, and crossing the two classes into an n x n matrix.

Normalization:
    If a normalization field is selected for a variable at a given time:
        normalized variable = variable / normalization field
"""

import arcpy
import math
import os
import numpy as np


# =================================================================================================
# TOOLBOX
# =================================================================================================

class Toolbox(object):

    def __init__(self):
        self.label = "GVSTMR"
        self.alias = "gvstmr"
        self.tools = [CalculateGVSTMR]


# =================================================================================================
# TOOL
# =================================================================================================

class CalculateGVSTMR(object):

    def __init__(self):

        self.label = "Calculate GVSTMR"

        self.description = (
            "Use multiple time-specific pairs of variables to calculate ONE "
            "period-integrated GVSTMR relationship map."
        )

    # =============================================================================================
    # PARAMETERS
    # =============================================================================================

    def getParameterInfo(self):

        # 0
        p0 = arcpy.Parameter(
            displayName="Input Polygon Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        p0.filter.list = ["Polygon"]

        # 1
        p1 = arcpy.Parameter(
            displayName="Number of Time Steps",
            name="number_of_times",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p1.value = 4
        p1.filter.type = "Range"
        p1.filter.list = [4, 50]

        # 2
        p2 = arcpy.Parameter(
            displayName="Variables and Optional Normalization Fields by Time",
            name="time_variable_table",
            datatype="GPValueTable",
            parameterType="Required",
            direction="Input"
        )
        p2.columns = [
            ["GPLong",   "Time", "ReadOnly"],
            ["GPString", "Variable 1"],
            ["GPString", "Normalization 1"],
            ["GPString", "Variable 2"],
            ["GPString", "Normalization 2"],
        ]
        for col in [1, 2, 3, 4]:
            p2.filters[col].type = "ValueList"
            p2.filters[col].list = []
        p2.values = [
            [1, None, "<None>", None, "<None>"],
            [2, None, "<None>", None, "<None>"],
            [3, None, "<None>", None, "<None>"],
            [4, None, "<None>", None, "<None>"],
        ]

        # 3
        p3 = arcpy.Parameter(
            displayName="Spatial Neighbors (K)",
            name="k_neighbors",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p3.value = 8
        p3.filter.type = "Range"
        p3.filter.list = [4, 500]

        # 4
        p4 = arcpy.Parameter(
            displayName="Classification Method",
            name="classification_method",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p4.filter.type = "ValueList"
        p4.filter.list = ["Quantile", "Equal Interval"]
        p4.value = "Quantile"

        # 5
        p5 = arcpy.Parameter(
            displayName="Grid Size",
            name="grid_size",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p5.filter.type = "ValueList"
        p5.filter.list = ["3 x 3", "5 x 5", "9 x 9"]
        p5.value = "3 x 3"

        # 6
        p6 = arcpy.Parameter(
            displayName="Apply GVSTMR Symbology",
            name="apply_symbology",
            datatype="GPBoolean",
            parameterType="Required",
            direction="Input"
        )
        p6.value = True

        # 7 — NEW
        p7 = arcpy.Parameter(
            displayName="Legend Style",
            name="legend_style",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p7.filter.type = "ValueList"
        p7.filter.list = [
            "Classified list",
            "Bivariate matrix"
        ]
        p7.value = "Bivariate matrix"

        # 8 — used only for native matrix legend
        p8 = arcpy.Parameter(
            displayName="Matrix Legend Template Layer",
            name="matrix_template_layer",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input"
        )
        p8.filter.list = ["Polygon"]

        # 9
        p9 = arcpy.Parameter(
            displayName="Output Feature Class",
            name="out_features",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]

    # =============================================================================================
    # DYNAMIC UI
    # =============================================================================================

    def updateParameters(self, parameters):

        p_in = parameters[0]
        p_n = parameters[1]
        p_tbl = parameters[2]

        try:
            n_times = int(p_n.value) if p_n.value is not None else 4
        except Exception:
            n_times = 4

        n_times = max(4, min(50, n_times))

        # Resize table while preserving current selections.
        old = p_tbl.values or []
        new_rows = []

        for i in range(n_times):

            if i < len(old):

                r = list(old[i])

                while len(r) < 5:
                    r.append(None)

                r[0] = i + 1

                if self._is_blank(r[2]):
                    r[2] = "<None>"

                if self._is_blank(r[4]):
                    r[4] = "<None>"

                new_rows.append(r[:5])

            else:

                new_rows.append(
                    [i + 1, None, "<None>", None, "<None>"]
                )

        if old != new_rows:
            p_tbl.values = new_rows

        # Numeric fields only.
        numeric_fields = []

        if p_in.valueAsText:

            try:

                allowed = {
                    "SmallInteger",
                    "Integer",
                    "Single",
                    "Double",
                    "BigInteger"
                }

                numeric_fields = [
                    f.name
                    for f in arcpy.ListFields(p_in.valueAsText)
                    if f.type in allowed
                ]

            except Exception:
                numeric_fields = []

        p_tbl.filters[1].list = numeric_fields
        p_tbl.filters[3].list = numeric_fields
        p_tbl.filters[2].list = ["<None>"] + numeric_fields
        p_tbl.filters[4].list = ["<None>"] + numeric_fields

        # Template selector only matters for matrix mode.
        legend_style = parameters[7].valueAsText
        parameters[8].enabled = (
            legend_style == "Bivariate matrix"
        )

        return

    # =============================================================================================
    # VALIDATION
    # =============================================================================================

    def updateMessages(self, parameters):

        p_in = parameters[0]
        p_n = parameters[1]
        p_tbl = parameters[2]
        p_k = parameters[3]

        try:
            n_times = int(p_n.value)
        except Exception:
            n_times = 4

        if n_times < 4:
            p_n.setErrorMessage(
                "At least four time steps are required for the temporal relationship."
            )

        rows = p_tbl.values or []

        if len(rows) != n_times:
            p_tbl.setErrorMessage(
                "The table must contain one row for every requested time step."
            )
            return

        missing = []

        for i, row in enumerate(rows):

            r = list(row) if row is not None else []

            v1 = r[1] if len(r) > 1 else None
            v2 = r[3] if len(r) > 3 else None

            if self._is_blank(v1) or self._is_blank(v2):
                missing.append("T{}".format(i + 1))

        if missing:
            p_tbl.setErrorMessage(
                "Variable 1 and Variable 2 are required for every time step. Missing: "
                + ", ".join(missing)
            )

        if p_in.valueAsText and p_k.value is not None:

            try:

                n_features = int(
                    arcpy.management.GetCount(p_in.valueAsText)[0]
                )

                if n_features < 5:
                    p_in.setErrorMessage(
                        "At least five polygons are required."
                    )

                elif int(p_k.value) >= n_features:
                    p_k.setWarningMessage(
                        "K is >= feature count and will be reduced to N - 1."
                    )

            except Exception:
                pass

        if (
            parameters[7].valueAsText == "Bivariate matrix"
            and
            not parameters[8].valueAsText
        ):
            parameters[8].setWarningMessage(
                "Select a polygon layer that already has ArcGIS Pro Bivariate Colors "
                "symbology if you want the native compact matrix legend. If omitted, "
                "the tool will try to auto-detect one in the active map."
            )

        return

    # =============================================================================================
    # EXECUTION
    # =============================================================================================

    def execute(self, parameters, messages):

        in_features = (
            parameters[0].valueAsText
        )

        n_times = int(
            parameters[1].value
        )

        table_rows = (
            parameters[2].values
            or []
        )

        k_neighbors = int(
            parameters[3].value
        )

        classification_method = (
            parameters[4].valueAsText
        )

        grid_text = (
            parameters[5].valueAsText
        )

        apply_symbology = bool(
            parameters[6].value
        )

        legend_style = (
            parameters[7].valueAsText
        )

        matrix_template = (
            parameters[8].valueAsText
        )

        out_features = (
            parameters[9].valueAsText
        )

        grid_n = int(
            grid_text.split(
                "x"
            )[0].strip()
        )

        # Store settings for postExecute.
        self._apply_after = (
            apply_symbology
        )

        self._grid_after = (
            grid_n
        )

        self._legend_style_after = (
            legend_style
        )

        self._matrix_template_after = (
            matrix_template
        )

        # -----------------------------------------------------------------------------------------
        # Parse time rows
        # -----------------------------------------------------------------------------------------

        specs = []

        for i in range(n_times):

            r = list(
                table_rows[i]
            )

            v1 = self._clean_field(
                r[1]
            )

            n1 = self._clean_field(
                r[2]
            )

            v2 = self._clean_field(
                r[3]
            )

            n2 = self._clean_field(
                r[4]
            )

            if not v1 or not v2:

                raise arcpy.ExecuteError(
                    "Variable 1 and Variable 2 are required for T{}."
                    .format(
                        i + 1
                    )
                )

            specs.append(
                {
                    "v1": v1,
                    "n1": n1,
                    "v2": v2,
                    "n2": n2
                }
            )

        arcpy.AddMessage(
            "GVSTMR: {} input times -> ONE period-integrated output."
            .format(
                n_times
            )
        )

        # -----------------------------------------------------------------------------------------
        # Copy input
        # -----------------------------------------------------------------------------------------

        arcpy.management.CopyFeatures(
            in_features,
            out_features
        )

        desc = arcpy.Describe(
            out_features
        )

        spatial_ref = (
            desc.spatialReference
        )

        n_features = int(
            arcpy.management.GetCount(
                out_features
            )[0]
        )

        if n_features < 5:

            raise arcpy.ExecuteError(
                "At least five polygons are required."
            )

        k_neighbors = min(
            k_neighbors,
            n_features - 1
        )

        # -----------------------------------------------------------------------------------------
        # Gather source fields
        # -----------------------------------------------------------------------------------------

        source_fields = []

        for s in specs:

            for field_name in [
                s["v1"],
                s["n1"],
                s["v2"],
                s["n2"]
            ]:

                if (
                    field_name
                    and
                    field_name not in source_fields
                ):

                    source_fields.append(
                        field_name
                    )

        existing_fields = {
            f.name: f
            for f in arcpy.ListFields(
                out_features
            )
        }

        for field_name in source_fields:

            if field_name not in existing_fields:

                raise arcpy.ExecuteError(
                    "Field not found: {}"
                    .format(
                        field_name
                    )
                )

        # -----------------------------------------------------------------------------------------
        # Read OIDs, centroids and values
        # -----------------------------------------------------------------------------------------

        cursor_fields = (
            [
                "OID@",
                "SHAPE@XY"
            ]
            +
            source_fields
        )

        oids = []
        xy_values = []
        attribute_rows = []

        with arcpy.da.SearchCursor(
            out_features,
            cursor_fields
        ) as cursor:

            for row in cursor:

                oids.append(
                    int(
                        row[0]
                    )
                )

                xy_values.append(
                    row[1]
                )

                attribute_rows.append(
                    row[2:]
                )

        oids = np.asarray(
            oids,
            dtype=np.int64
        )

        xy = np.asarray(
            xy_values,
            dtype=float
        )

        raw = np.asarray(
            attribute_rows,
            dtype=object
        )

        field_index = {
            field_name: idx
            for idx, field_name
            in enumerate(
                source_fields
            )
        }

        # -----------------------------------------------------------------------------------------
        # Build time x polygon X/Y matrices
        # -----------------------------------------------------------------------------------------

        X = np.full(
            (
                n_times,
                n_features
            ),
            np.nan,
            dtype=float
        )

        Y = np.full(
            (
                n_times,
                n_features
            ),
            np.nan,
            dtype=float
        )

        for t, s in enumerate(
            specs
        ):

            x = self._numeric_column(
                raw[
                    :,
                    field_index[
                        s["v1"]
                    ]
                ]
            )

            y = self._numeric_column(
                raw[
                    :,
                    field_index[
                        s["v2"]
                    ]
                ]
            )

            if s["n1"]:

                denominator = self._numeric_column(
                    raw[
                        :,
                        field_index[
                            s["n1"]
                        ]
                    ]
                )

                x = np.divide(
                    x,
                    denominator,
                    out=np.full_like(
                        x,
                        np.nan
                    ),
                    where=(
                        np.isfinite(
                            x
                        )
                        &
                        np.isfinite(
                            denominator
                        )
                        &
                        (
                            denominator != 0
                        )
                    )
                )

            if s["n2"]:

                denominator = self._numeric_column(
                    raw[
                        :,
                        field_index[
                            s["n2"]
                        ]
                    ]
                )

                y = np.divide(
                    y,
                    denominator,
                    out=np.full_like(
                        y,
                        np.nan
                    ),
                    where=(
                        np.isfinite(
                            y
                        )
                        &
                        np.isfinite(
                            denominator
                        )
                        &
                        (
                            denominator != 0
                        )
                    )
                )

            X[
                t,
                :
            ] = x

            Y[
                t,
                :
            ] = y

        # -----------------------------------------------------------------------------------------
        # Spatial neighborhoods
        # -----------------------------------------------------------------------------------------

        arcpy.AddMessage(
            "Calculating period-integrated spatial relationship..."
        )

        coords = self._coordinates_for_knn(
            xy,
            spatial_ref
        )

        nbr = self._knn_indices_excluding_self(
            coords,
            k_neighbors
        )

        spatial_min_valid_neighbors = min(
            k_neighbors,
            max(
                4,
                int(
                    math.ceil(
                        0.60
                        *
                        k_neighbors
                    )
                )
            )
        )

        # covariance components for each time
        vx_s = np.full(
            (
                n_times,
                n_features
            ),
            np.nan,
            dtype=float
        )

        vy_s = np.full(
            (
                n_times,
                n_features
            ),
            np.nan,
            dtype=float
        )

        cov_s = np.full(
            (
                n_times,
                n_features
            ),
            np.nan,
            dtype=float
        )

        for t in range(n_times):

            local_x = X[
                t,
                :
            ][
                nbr
            ]

            local_y = Y[
                t,
                :
            ][
                nbr
            ]

            vx, vy, cv = (
                self._row_cov_components(
                    local_x,
                    local_y,
                    min_n=spatial_min_valid_neighbors
                )
            )

            vx_s[
                t,
                :
            ] = vx

            vy_s[
                t,
                :
            ] = vy

            cov_s[
                t,
                :
            ] = cv

        # Period integration = average covariance matrices through time.
        valid_spatial_time = (
            np.isfinite(
                vx_s
            )
            &
            np.isfinite(
                vy_s
            )
            &
            np.isfinite(
                cov_s
            )
        )

        n_valid_spatial_times = (
            valid_spatial_time.sum(
                axis=0
            )
        )

        with np.errstate(
            invalid="ignore"
        ):

            mean_vx_s = np.nanmean(
                vx_s,
                axis=0
            )

            mean_vy_s = np.nanmean(
                vy_s,
                axis=0
            )

            mean_cov_s = np.nanmean(
                cov_s,
                axis=0
            )

        rho_s = self._rho_from_cov(
            mean_vx_s,
            mean_vy_s,
            mean_cov_s
        )

        # Require spatial covariance information for most requested times.
        min_spatial_times = max(
            2,
            int(
                math.ceil(
                    0.75
                    *
                    n_times
                )
            )
        )

        rho_s[
            n_valid_spatial_times
            <
            min_spatial_times
        ] = np.nan

        # -----------------------------------------------------------------------------------------
        # Temporal relationship: ALL requested times in the focal polygon
        # -----------------------------------------------------------------------------------------

        arcpy.AddMessage(
            "Calculating full-period temporal relationship..."
        )

        temporal_min_valid = max(
            4,
            int(
                math.ceil(
                    0.75
                    *
                    n_times
                )
            )
        )

        temporal_min_valid = min(
            temporal_min_valid,
            n_times
        )

        vx_t, vy_t, cov_t = (
            self._row_cov_components(
                X.T,
                Y.T,
                min_n=temporal_min_valid
            )
        )

        rho_t = self._rho_from_cov(
            vx_t,
            vy_t,
            cov_t
        )

        # -----------------------------------------------------------------------------------------
        # ONE classification across polygons
        # -----------------------------------------------------------------------------------------

        valid_pair = (
            np.isfinite(
                rho_s
            )
            &
            np.isfinite(
                rho_t
            )
        )

        arcpy.AddMessage(
            "Valid final spatial-temporal pairs: {:,}/{:,}"
            .format(
                int(
                    valid_pair.sum()
                ),
                n_features
            )
        )

        if not np.any(
            valid_pair
        ):

            raise arcpy.ExecuteError(
                "No polygons have both valid spatial and temporal GVSTMR relationships."
            )

        spatial_edges = self._classification_edges(
            rho_s,
            grid_n,
            classification_method
        )

        temporal_edges = self._classification_edges(
            rho_t,
            grid_n,
            classification_method
        )

        arcpy.AddMessage(
            "Spatial rho^S breaks: {}"
            .format(
                self._format_edges(
                    spatial_edges
                )
            )
        )

        arcpy.AddMessage(
            "Temporal rho^T breaks: {}"
            .format(
                self._format_edges(
                    temporal_edges
                )
            )
        )

        spatial_class = self._classify(
            rho_s,
            spatial_edges,
            grid_n
        )

        temporal_class = self._classify(
            rho_t,
            temporal_edges,
            grid_n
        )

        gvstmr = np.full(
            n_features,
            -1,
            dtype=np.int16
        )

        valid = (
            (
                spatial_class > 0
            )
            &
            (
                temporal_class > 0
            )
        )

        gvstmr[
            valid
        ] = (
            (
                spatial_class[
                    valid
                ]
                -
                1
            )
            *
            grid_n
            +
            temporal_class[
                valid
            ]
        ).astype(
            np.int16
        )

        # -----------------------------------------------------------------------------------------
        # ONE output set
        # -----------------------------------------------------------------------------------------

        self._add_field(
            out_features,
            "RHO_S",
            "DOUBLE"
        )

        self._add_field(
            out_features,
            "RHO_T",
            "DOUBLE"
        )

        self._add_field(
            out_features,
            "GVSTMR",
            "SHORT"
        )

        oid_to_idx = {
            int(
                oid
            ): idx
            for idx, oid in enumerate(
                oids
            )
        }

        with arcpy.da.UpdateCursor(
            out_features,
            [
                "OID@",
                "RHO_S",
                "RHO_T",
                "GVSTMR"
            ]
        ) as cursor:

            for row in cursor:

                idx = oid_to_idx[
                    int(
                        row[0]
                    )
                ]

                s_val = rho_s[
                    idx
                ]

                t_val = rho_t[
                    idx
                ]

                g_val = int(
                    gvstmr[
                        idx
                    ]
                )

                row[
                    1
                ] = (
                    float(
                        s_val
                    )
                    if np.isfinite(
                        s_val
                    )
                    else None
                )

                row[
                    2
                ] = (
                    float(
                        t_val
                    )
                    if np.isfinite(
                        t_val
                    )
                    else None
                )

                row[
                    3
                ] = (
                    g_val
                    if g_val > 0
                    else None
                )

                cursor.updateRow(
                    row
                )

        arcpy.AddMessage(
            "GVSTMR complete: multiple times -> one RHO_S, one RHO_T, one GVSTMR field."
        )

        return

    # =============================================================================================
    # POST EXECUTE — FORCE OUTPUT TO GVSTMR UNIQUE-VALUE SYMBOLOGY
    # =============================================================================================

    def postExecute(self, parameters):

        """
        Legend Style:
            Classified list
                -> GVSTMR unique-value classes.

            Bivariate matrix
                -> native ArcGIS Pro Bivariate Colors renderer:
                   vertical   = RHO_S
                   horizontal = RHO_T

        The analytical calculation does not change between these modes.
        """

        try:

            if not bool(parameters[6].value):
                return

            legend_style = parameters[7].valueAsText
            template_value = parameters[8].valueAsText
            out_path = parameters[9].valueAsText

            grid_text = parameters[5].valueAsText
            grid_n = int(grid_text.split("x")[0].strip())

            classification_method = parameters[4].valueAsText

            project = arcpy.mp.ArcGISProject("CURRENT")
            active_map = project.activeMap

            if active_map is None:
                return

            output_layer = self._find_output_layer(
                active_map,
                out_path
            )

            if output_layer is None:
                output_layer = active_map.addDataFromPath(
                    out_path
                )

            # -------------------------------------------------------------------------------------
            # OPTION A — normal class-list legend
            # -------------------------------------------------------------------------------------

            if legend_style == "Classified list":

                self._apply_unique_value_fallback(
                    output_layer,
                    grid_n
                )

                arcpy.AddMessage(
                    "Legend Style = Classified list."
                )

                return

            # -------------------------------------------------------------------------------------
            # OPTION B — native matrix legend
            # -------------------------------------------------------------------------------------

            if grid_n != 3:

                self._apply_unique_value_fallback(
                    output_layer,
                    grid_n
                )

                arcpy.AddWarning(
                    "The native ArcGIS Pro Bivariate Colors matrix legend is available "
                    "for 2x2, 3x3 and 4x4. The GVSTMR calculation is still correct, "
                    "but this toolbox falls back to a class-list legend for {}."
                    .format(grid_text)
                )

                return

            template_layer = self._resolve_bivariate_template(
                active_map,
                template_value,
                output_layer
            )

            if template_layer is None:

                self._apply_unique_value_fallback(
                    output_layer,
                    grid_n
                )

                arcpy.AddWarning(
                    "No native Bivariate Colors template layer was found. "
                    "The tool safely used the classified-list legend instead. "
                    "For the matrix legend, select any polygon layer already using "
                    "ArcGIS Pro Bivariate Colors."
                )

                return

            import copy

            template_cim = template_layer.getDefinition(
                "V3"
            )

            template_renderer = template_cim.renderer

            authoring = getattr(
                template_renderer,
                "authoringInfo",
                None
            )

            field_infos = getattr(
                authoring,
                "fieldInfos",
                None
            )

            if (
                authoring is None
                or field_infos is None
                or len(field_infos) < 2
            ):
                raise RuntimeError(
                    "The selected template is not a native ArcGIS Pro Bivariate Colors layer."
                )

            source_field_1 = field_infos[0].field
            source_field_2 = field_infos[1].field

            # Copy the real native bivariate renderer from the template.
            renderer = copy.deepcopy(
                template_renderer
            )

            # Replace every persisted template-field reference.
            renderer = self._replace_cim_strings(
                renderer,
                {
                    source_field_1: "RHO_S",
                    source_field_2: "RHO_T"
                }
            )

            # Explicitly set commonly persisted renderer properties.
            try:
                renderer.fields = [
                    "RHO_S",
                    "RHO_T"
                ]
            except Exception:
                pass

            authoring = getattr(
                renderer,
                "authoringInfo",
                None
            )

            field_infos = getattr(
                authoring,
                "fieldInfos",
                None
            )

            # Recalculate the same target-data breaks used for the matrix display.
            rho_s_values = []
            rho_t_values = []

            with arcpy.da.SearchCursor(
                out_path,
                ["RHO_S", "RHO_T"]
            ) as cursor:

                for row in cursor:

                    try:
                        if (
                            row[0] is not None
                            and math.isfinite(float(row[0]))
                        ):
                            rho_s_values.append(float(row[0]))
                    except Exception:
                        pass

                    try:
                        if (
                            row[1] is not None
                            and math.isfinite(float(row[1]))
                        ):
                            rho_t_values.append(float(row[1]))
                    except Exception:
                        pass

            spatial_edges = self._classification_edges(
                np.asarray(rho_s_values, dtype=float),
                3,
                classification_method
            )

            temporal_edges = self._classification_edges(
                np.asarray(rho_t_values, dtype=float),
                3,
                classification_method
            )

            if field_infos is not None and len(field_infos) >= 2:

                field_infos[0].field = "RHO_S"
                field_infos[1].field = "RHO_T"

                cim_method = (
                    "Quantile"
                    if classification_method == "Quantile"
                    else "EqualInterval"
                )

                for info in field_infos[:2]:

                    try:
                        info.classificationMethod = cim_method
                    except Exception:
                        pass

                try:
                    field_infos[0].minimumBreak = float(spatial_edges[0])
                    field_infos[0].upperBounds = [
                        float(x)
                        for x in spatial_edges[1:]
                    ]
                except Exception:
                    pass

                try:
                    field_infos[1].minimumBreak = float(temporal_edges[0])
                    field_infos[1].upperBounds = [
                        float(x)
                        for x in temporal_edges[1:]
                    ]
                except Exception:
                    pass

                try:
                    field_infos[0].defaultLabel = "Spatial relationship"
                    field_infos[1].defaultLabel = "Temporal relationship"
                except Exception:
                    pass

            try:
                authoring.gridSize = "ThreeByThree"
            except Exception:
                pass

            try:
                authoring.gridLabelOption = "Sides"
            except Exception:
                pass

            # Replace output renderer with the cloned native bivariate renderer.
            output_cim = output_layer.getDefinition(
                "V3"
            )

            output_cim.renderer = renderer

            output_layer.setDefinition(
                output_cim
            )

            # User-friendly field aliases.
            try:
                arcpy.management.AlterField(
                    out_path,
                    "RHO_S",
                    new_field_alias="Spatial relationship rhoS"
                )
            except Exception:
                pass

            try:
                arcpy.management.AlterField(
                    out_path,
                    "RHO_T",
                    new_field_alias="Temporal relationship rhoT"
                )
            except Exception:
                pass

            arcpy.AddMessage(
                "Legend Style = Bivariate matrix "
                "(vertical RHO_S × horizontal RHO_T)."
            )

        except Exception as ex:

            # Safe fallback: never leave the output with a broken single symbol.
            try:

                project = arcpy.mp.ArcGISProject("CURRENT")
                active_map = project.activeMap
                out_path = parameters[9].valueAsText

                output_layer = self._find_output_layer(
                    active_map,
                    out_path
                )

                if output_layer is not None:

                    grid_text = parameters[5].valueAsText
                    grid_n = int(
                        grid_text.split("x")[0].strip()
                    )

                    self._apply_unique_value_fallback(
                        output_layer,
                        grid_n
                    )

                arcpy.AddWarning(
                    "GVSTMR calculation succeeded. Matrix legend application failed, "
                    "so the tool reverted to the classified-list legend: {}"
                    .format(ex)
                )

            except Exception:
                pass

        return

    # =============================================================================================
    # LEGEND HELPERS
    # =============================================================================================

    @staticmethod
    def _find_output_layer(
        active_map,
        out_path
    ):

        if active_map is None:
            return None

        try:

            target_path = os.path.normcase(
                os.path.normpath(
                    arcpy.Describe(out_path).catalogPath
                )
            )

        except Exception:
            return None

        for layer in active_map.listLayers():

            if not layer.isFeatureLayer:
                continue

            try:

                layer_path = os.path.normcase(
                    os.path.normpath(
                        arcpy.Describe(layer).catalogPath
                    )
                )

                if layer_path == target_path:
                    return layer

            except Exception:
                continue

        return None

    @classmethod
    def _resolve_bivariate_template(
        cls,
        active_map,
        template_value,
        output_layer
    ):

        # Explicit user selection first.
        if template_value:

            try:
                wanted_desc = arcpy.Describe(
                    template_value
                )
                wanted_name = wanted_desc.name
            except Exception:
                wanted_name = str(template_value)

            for layer in active_map.listLayers():

                if not layer.isFeatureLayer:
                    continue

                try:

                    if (
                        layer.name == wanted_name
                        or layer.name == str(template_value)
                    ):
                        if cls._is_native_bivariate_layer(layer):
                            return layer

                except Exception:
                    continue

        # Otherwise auto-detect another native bivariate layer already in the map.
        for layer in active_map.listLayers():

            if not layer.isFeatureLayer:
                continue

            if layer == output_layer:
                continue

            try:
                if cls._is_native_bivariate_layer(layer):
                    return layer
            except Exception:
                continue

        return None

    @staticmethod
    def _is_native_bivariate_layer(layer):

        try:

            cim = layer.getDefinition(
                "V3"
            )

            renderer = getattr(
                cim,
                "renderer",
                None
            )

            if renderer is None:
                return False

            authoring = getattr(
                renderer,
                "authoringInfo",
                None
            )

            if authoring is None:
                return False

            field_infos = getattr(
                authoring,
                "fieldInfos",
                None
            )

            grid_size = getattr(
                authoring,
                "gridSize",
                None
            )

            return (
                field_infos is not None
                and len(field_infos) >= 2
                and grid_size is not None
            )

        except Exception:
            return False

    @classmethod
    def _replace_cim_strings(
        cls,
        obj,
        replacements,
        visited=None
    ):

        """
        Recursively replace source-field names inside a cloned CIM renderer.
        This catches fields, authoring metadata and persisted expressions.
        """

        if visited is None:
            visited = set()

        if obj is None:
            return obj

        if isinstance(obj, str):

            result = obj

            for old, new in replacements.items():

                if old:
                    result = result.replace(
                        old,
                        new
                    )

            return result

        if isinstance(
            obj,
            (int, float, bool, bytes)
        ):
            return obj

        object_id = id(obj)

        if object_id in visited:
            return obj

        visited.add(object_id)

        if isinstance(obj, list):

            for i in range(len(obj)):

                obj[i] = cls._replace_cim_strings(
                    obj[i],
                    replacements,
                    visited
                )

            return obj

        if isinstance(obj, tuple):

            return tuple(
                cls._replace_cim_strings(
                    value,
                    replacements,
                    visited
                )
                for value in obj
            )

        if isinstance(obj, dict):

            for key in list(obj.keys()):

                obj[key] = cls._replace_cim_strings(
                    obj[key],
                    replacements,
                    visited
                )

            return obj

        if hasattr(obj, "__dict__"):

            for name, value in list(
                vars(obj).items()
            ):

                try:

                    new_value = cls._replace_cim_strings(
                        value,
                        replacements,
                        visited
                    )

                    setattr(
                        obj,
                        name,
                        new_value
                    )

                except Exception:
                    pass

        return obj

    @classmethod
    def _apply_unique_value_fallback(
        cls,
        output_layer,
        grid_n
    ):

        sym = output_layer.symbology

        sym.updateRenderer(
            "UniqueValueRenderer"
        )

        sym.renderer.fields = [
            "GVSTMR"
        ]

        output_layer.symbology = sym

        sym = output_layer.symbology

        for group in sym.renderer.groups:

            for item in group.items:

                raw = cls._unwrap_unique_value(
                    item.values
                )

                try:
                    cell = int(float(raw))
                except Exception:
                    continue

                if (
                    cell < 1
                    or cell > grid_n * grid_n
                ):
                    continue

                spatial_level = (
                    ((cell - 1) // grid_n) + 1
                )

                temporal_level = (
                    ((cell - 1) % grid_n) + 1
                )

                red, green, blue = cls._matrix_rgb(
                    spatial_level,
                    temporal_level,
                    grid_n
                )

                try:
                    item.symbol.color = {
                        "RGB": [
                            red,
                            green,
                            blue,
                            100
                        ]
                    }
                except Exception:
                    pass

                item.label = cls._class_label(
                    spatial_level,
                    temporal_level,
                    grid_n
                )

        output_layer.symbology = sym

    # =============================================================================================
    # NUMERIC HELPERS
    # =============================================================================================

    @staticmethod
    def _is_blank(value):

        if value is None:
            return True

        return str(
            value
        ).strip() in (
            "",
            "#",
            "None",
            "<None>"
        )

    @classmethod
    def _clean_field(
        cls,
        value
    ):

        if cls._is_blank(
            value
        ):
            return None

        return str(
            value
        ).strip()

    @staticmethod
    def _numeric_column(values):

        out = np.full(
            len(
                values
            ),
            np.nan,
            dtype=float
        )

        for i, value in enumerate(
            values
        ):

            if value is None:
                continue

            try:

                number = float(
                    value
                )

                if math.isfinite(
                    number
                ):

                    out[
                        i
                    ] = number

            except Exception:

                pass

        return out

    @staticmethod
    def _coordinates_for_knn(
        xy,
        spatial_ref
    ):

        xy = np.asarray(
            xy,
            dtype=float
        )

        try:

            is_geographic = (
                str(
                    spatial_ref.type
                ).lower()
                ==
                "geographic"
            )

        except Exception:

            is_geographic = False

        if not is_geographic:
            return xy

        lon = np.deg2rad(
            xy[
                :,
                0
            ]
        )

        lat = np.deg2rad(
            xy[
                :,
                1
            ]
        )

        lat0 = np.nanmean(
            lat
        )

        earth_radius_km = (
            6371.0088
        )

        return np.column_stack(
            [
                earth_radius_km
                *
                lon
                *
                np.cos(
                    lat0
                ),

                earth_radius_km
                *
                lat
            ]
        )

    @staticmethod
    def _knn_indices_excluding_self(
        coords,
        k
    ):

        coords = np.asarray(
            coords,
            dtype=float
        )

        try:

            from scipy.spatial import cKDTree

            tree = cKDTree(
                coords
            )

            _, idx = tree.query(
                coords,
                k=k + 1
            )

            if idx.ndim == 1:

                idx = idx[
                    :,
                    None
                ]

            return idx[
                :,
                1:k + 1
            ]

        except Exception:

            arcpy.AddWarning(
                "SciPy unavailable; using NumPy KNN fallback."
            )

        n = len(
            coords
        )

        result = np.empty(
            (
                n,
                k
            ),
            dtype=np.int64
        )

        block = 256

        for start in range(
            0,
            n,
            block
        ):

            end = min(
                n,
                start + block
            )

            q = coords[
                start:end
            ]

            dist2 = (
                (
                    q[
                        :,
                        None,
                        0
                    ]
                    -
                    coords[
                        None,
                        :,
                        0
                    ]
                ) ** 2
                +
                (
                    q[
                        :,
                        None,
                        1
                    ]
                    -
                    coords[
                        None,
                        :,
                        1
                    ]
                ) ** 2
            )

            row_ids = np.arange(
                start,
                end
            )

            dist2[
                np.arange(
                    end - start
                ),
                row_ids
            ] = np.inf

            selected = np.argpartition(
                dist2,
                kth=k - 1,
                axis=1
            )[
                :,
                :k
            ]

            selected_dist = np.take_along_axis(
                dist2,
                selected,
                axis=1
            )

            order = np.argsort(
                selected_dist,
                axis=1
            )

            result[
                start:end,
                :
            ] = np.take_along_axis(
                selected,
                order,
                axis=1
            )

        return result

    @staticmethod
    def _row_cov_components(
        x,
        y,
        min_n
    ):

        x = np.asarray(
            x,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=float
        )

        valid = (
            np.isfinite(
                x
            )
            &
            np.isfinite(
                y
            )
        )

        n = valid.sum(
            axis=1
        )

        xx = np.where(
            valid,
            x,
            0.0
        )

        yy = np.where(
            valid,
            y,
            0.0
        )

        mean_x = np.divide(
            xx.sum(
                axis=1
            ),
            n,
            out=np.full(
                len(
                    n
                ),
                np.nan
            ),
            where=(
                n > 0
            )
        )

        mean_y = np.divide(
            yy.sum(
                axis=1
            ),
            n,
            out=np.full(
                len(
                    n
                ),
                np.nan
            ),
            where=(
                n > 0
            )
        )

        dx = np.where(
            valid,
            x
            -
            mean_x[
                :,
                None
            ],
            0.0
        )

        dy = np.where(
            valid,
            y
            -
            mean_y[
                :,
                None
            ],
            0.0
        )

        denominator = np.maximum(
            n - 1,
            1
        )

        var_x = (
            dx
            *
            dx
        ).sum(
            axis=1
        ) / denominator

        var_y = (
            dy
            *
            dy
        ).sum(
            axis=1
        ) / denominator

        covariance = (
            dx
            *
            dy
        ).sum(
            axis=1
        ) / denominator

        invalid = (
            n < min_n
        )

        var_x[
            invalid
        ] = np.nan

        var_y[
            invalid
        ] = np.nan

        covariance[
            invalid
        ] = np.nan

        return (
            var_x,
            var_y,
            covariance
        )

    @staticmethod
    def _rho_from_cov(
        var_x,
        var_y,
        covariance
    ):

        denominator = np.sqrt(
            var_x
            *
            var_y
        )

        rho = np.divide(
            covariance,
            denominator,
            out=np.full(
                len(
                    denominator
                ),
                np.nan
            ),
            where=(
                np.isfinite(
                    denominator
                )
                &
                (
                    denominator > 1e-15
                )
            )
        )

        return np.clip(
            rho,
            -1.0,
            1.0
        )

    @staticmethod
    def _classification_edges(
        values,
        n_classes,
        method
    ):

        data = np.asarray(
            values,
            dtype=float
        )

        data = data[
            np.isfinite(
                data
            )
        ]

        if len(
            data
        ) == 0:

            raise arcpy.ExecuteError(
                "No valid values are available for GVSTMR classification."
            )

        if method == "Equal Interval":

            edges = np.linspace(
                -1.0,
                1.0,
                n_classes + 1
            )

        else:

            edges = np.quantile(
                data,
                np.linspace(
                    0.0,
                    1.0,
                    n_classes + 1
                )
            )

            edges[
                0
            ] = -1.0

            edges[
                -1
            ] = 1.0

            eps = 1e-10

            for i in range(
                1,
                len(
                    edges
                )
                -
                1
            ):

                if (
                    edges[
                        i
                    ]
                    <=
                    edges[
                        i - 1
                    ]
                ):

                    edges[
                        i
                    ] = (
                        edges[
                            i - 1
                        ]
                        +
                        eps
                    )

            for i in range(
                len(
                    edges
                )
                -
                2,
                0,
                -1
            ):

                if (
                    edges[
                        i
                    ]
                    >=
                    edges[
                        i + 1
                    ]
                ):

                    edges[
                        i
                    ] = (
                        edges[
                            i + 1
                        ]
                        -
                        eps
                    )

        return np.asarray(
            edges,
            dtype=float
        )

    @staticmethod
    def _classify(
        values,
        edges,
        n_classes
    ):

        values = np.asarray(
            values,
            dtype=float
        )

        out = np.full(
            values.shape,
            -1,
            dtype=np.int16
        )

        valid = np.isfinite(
            values
        )

        internal = np.asarray(
            edges[
                1:-1
            ],
            dtype=float
        )

        out[
            valid
        ] = (
            np.searchsorted(
                internal,
                values[
                    valid
                ],
                side="right"
            )
            +
            1
        ).astype(
            np.int16
        )

        out[
            (
                out > n_classes
            )
            &
            valid
        ] = n_classes

        return out

    @staticmethod
    def _add_field(
        dataset,
        field_name,
        field_type
    ):

        existing = {
            f.name.upper()
            for f in arcpy.ListFields(
                dataset
            )
        }

        if (
            field_name.upper()
            not in existing
        ):

            arcpy.management.AddField(
                dataset,
                field_name,
                field_type
            )

    @staticmethod
    def _format_edges(edges):

        return (
            "["
            +
            ", ".join(
                "{:.4f}".format(
                    float(
                        x
                    )
                )
                for x in edges
            )
            +
            "]"
        )

    # =============================================================================================
    # SYMBOLOGY
    # =============================================================================================

    @staticmethod
    def _hex_to_rgb(value):

        value = value.lstrip(
            "#"
        )

        return tuple(
            int(
                value[
                    i:i + 2
                ],
                16
            )
            for i in (
                0,
                2,
                4
            )
        )

    @classmethod
    def _matrix_rgb(
        cls,
        spatial_level,
        temporal_level,
        n
    ):

        exact_3 = {
            1: "#E5E4E9",
            2: "#B9D9E6",
            3: "#54B8D0",
            4: "#DB95CB",
            5: "#9E9FCB",
            6: "#4582BB",
            7: "#BE3F98",
            8: "#74529F",
            9: "#243A83"
        }

        if n == 3:

            cell = (
                (
                    spatial_level
                    -
                    1
                )
                *
                3
                +
                temporal_level
            )

            return cls._hex_to_rgb(
                exact_3[
                    cell
                ]
            )

        low_low = np.asarray(
            cls._hex_to_rgb(
                "#E5E4E9"
            ),
            dtype=float
        )

        low_high = np.asarray(
            cls._hex_to_rgb(
                "#54B8D0"
            ),
            dtype=float
        )

        high_low = np.asarray(
            cls._hex_to_rgb(
                "#BE3F98"
            ),
            dtype=float
        )

        high_high = np.asarray(
            cls._hex_to_rgb(
                "#243A83"
            ),
            dtype=float
        )

        temporal_fraction = (
            float(
                temporal_level - 1
            )
            /
            float(
                n - 1
            )
        )

        spatial_fraction = (
            float(
                spatial_level - 1
            )
            /
            float(
                n - 1
            )
        )

        rgb = (
            (
                1
                -
                temporal_fraction
            )
            *
            (
                1
                -
                spatial_fraction
            )
            *
            low_low
            +
            temporal_fraction
            *
            (
                1
                -
                spatial_fraction
            )
            *
            low_high
            +
            (
                1
                -
                temporal_fraction
            )
            *
            spatial_fraction
            *
            high_low
            +
            temporal_fraction
            *
            spatial_fraction
            *
            high_high
        )

        return tuple(
            int(
                round(
                    x
                )
            )
            for x in rgb
        )

    @staticmethod
    def _unwrap_unique_value(values):

        value = values

        try:

            while (
                isinstance(
                    value,
                    (
                        list,
                        tuple
                    )
                )
                and
                len(
                    value
                )
                >
                0
            ):

                value = value[
                    0
                ]

        except Exception:

            return None

        return value

    @staticmethod
    def _class_label(
        spatial_level,
        temporal_level,
        n
    ):

        if n == 3:

            names = {
                1: "Low",
                2: "Medium",
                3: "High"
            }

            return (
                "{} spatial / {} temporal"
                .format(
                    names[
                        spatial_level
                    ],
                    names[
                        temporal_level
                    ]
                )
            )

        return (
            "Spatial {}/{} / Temporal {}/{}"
            .format(
                spatial_level,
                n,
                temporal_level,
                n
            )
        )
