"""
Analysis utilities for MuVIcell factor interpretation.

This module provides comprehensive functions for analyzing and interpreting MuVI factors,
including variance analysis, feature characterization, and factor associations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple, Callable
import warnings

# Statistical tests
from scipy.stats import kruskal, kendalltau


def _to_factor_labels(names: Sequence[str]) -> List[str]:
    """
    Map ['factor_0','factor_1', ...] to ['Factor 1','Factor 2', ...] if pattern matches,
    otherwise return names unchanged.
    """
    out = []
    for n in names:
        try:
            if isinstance(n, str) and "factor_" in n:
                idx = int(n.split("_")[1])
                out.append(f"Factor {idx+1}")
            else:
                out.append(n)
        except Exception:
            out.append(n)
    return out


def _rename_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename factor_* columns to human readable Factor i."""
    renamed = df.copy()
    newcols = []
    for c in renamed.columns:
        if isinstance(c, str) and c.startswith("factor_"):
            try:
                newcols.append(f"Factor {int(c.split('_')[1]) + 1}")
            except Exception:
                newcols.append(c)
        else:
            newcols.append(c)
    renamed.columns = newcols
    return renamed


def _nan_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation with NaN handling."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]


def _factor_names_from_model(model) -> List[str]:
    """Get factor names from model."""
    try:
        fs = model.get_factor_scores(as_df=True)
        cols = list(fs.columns)
        return cols
    except Exception:
        try:
            k = model.n_factors
        except Exception:
            k = 10
        return [f"factor_{i}" for i in range(k)]


def _view_names(model, mdata) -> List[str]:
    """Get view names from model or mdata."""
    for attr in ["view_names", "views", "modalities"]:
        if hasattr(model, attr):
            v = getattr(model, attr)
            return list(v)
    return list(mdata.mod.keys())


def muvi_reconstruction_info(model, mdata, views: Optional[Sequence[str]] = None, verbosity: int = 1) -> dict:
    """
    Compute R and R2 between original X and reconstructed X = scores @ loadings, per view.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    mdata : MuData
        Multi-view data object
    views : list, optional
        Views to analyze. If None, analyzes all views
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    dict
        Dictionary with keys:
        'by_view': DataFrame[view, R, R2]
        'macro': {'R': macro_R, 'R2': macro_R2}
    """
    if views is None:
        views = _view_names(model, mdata)

    # scores: (n_obs x n_factors)
    scores = model.get_factor_scores()
    if isinstance(scores, pd.DataFrame):
        scores = scores.to_numpy()

    per_view = []
    loadings = model.get_factor_loadings()  # dict or similar keyed by view
    for v in views:
        L = loadings[v]  # shape: n_factors x n_features_v
        # rebuild
        rec = scores @ L  # (n_obs x n_features_v)
        X = mdata[v].X  # original (n_obs x n_features_v)
        r = _nan_pearsonr(np.asarray(X).ravel(), np.asarray(rec).ravel())
        per_view.append({"view": v, "R": r, "R2": None if pd.isna(r) else r * r})

    df = pd.DataFrame(per_view)
    macro_R = df["R"].mean(skipna=True)
    macro_R2 = df["R2"].mean(skipna=True)

    if verbosity:
        print("Reconstruction macro R:", np.round(macro_R, 3))
        print("Reconstruction macro R2:", np.round(macro_R2, 3))
        print(df.sort_values("R2", ascending=False).to_string(index=False))

    return {"by_view": df, "macro": {"R": macro_R, "R2": macro_R2}}


def muvi_variance_by_view_info(model, view_name_transform: Optional[Callable[[str], str]] = None,
                               verbosity: int = 1) -> pd.DataFrame:
    """
    Get variance explained by factors per view.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    view_name_transform : callable, optional
        Function to transform view names
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Long tidy DataFrame with columns [Factor, View, Variance]
    """
    vexp = __import__("muvi").tl.variance_explained(model)[1]  # DataFrame factors x views
    df = vexp.copy()
    df.index.name = "Factor"
    df = df.reset_index().melt(id_vars="Factor", var_name="View", value_name="Variance")

    # Human friendly factor labels where possible
    df["Factor"] = _to_factor_labels(df["Factor"])
    # Optional view transformation
    if view_name_transform is not None:
        df["View"] = df["View"].map(view_name_transform)

    # Ordered category by factor index if parsable
    facs = pd.unique(df["Factor"])
    df["Factor"] = pd.Categorical(df["Factor"], categories=facs, ordered=True)

    if verbosity:
        print(df.head().to_string(index=False))
    return df


def muvi_featureclass_variance_info(model, mdata,
                                    feature_type_map: Dict[str, List[str]],
                                    aggregator: str = "median",
                                    verbosity: int = 1) -> pd.DataFrame:
    """
    Compute variance explained per factor for each feature class across views.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    mdata : MuData
        Multi-view data object
    feature_type_map : dict
        Dictionary mapping class names to feature lists
    aggregator : str, default 'median'
        Aggregation method across views ('median' or 'mean')
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Long DataFrame [Factor, Feature_type, Variance]
    """
    assert aggregator in {"median", "mean"}
    views = _view_names(model, mdata)
    import muvi as _muvi  # local import to avoid collisions

    rows = []
    for cls, feats in feature_type_map.items():
        per_view_vals = []
        for v in views:
            feats_in_view = [f for f in feats if f in list(mdata[v].var_names)]
            if len(feats_in_view) == 0:
                continue
            # r2[1] is per-factor variance for the selected features in this view
            r2 = _muvi.tl.variance_explained(model, view_idx=v, feature_idx=feats_in_view, cache=False, sort=False)[1]
            per_view_vals.append(r2.values.reshape(-1, 1))  # factors x 1

        if len(per_view_vals) == 0:
            continue

        A = np.hstack(per_view_vals)  # factors x n_views_present
        agg = np.median(A, axis=1) if aggregator == "median" else np.mean(A, axis=1)
        # Build factor names from r2.index
        factor_labels = _to_factor_labels(list(r2.index))
        rows.extend([{"Factor": f, "Feature_type": cls, "Variance": val} for f, val in zip(factor_labels, agg)])

    out = pd.DataFrame(rows)
    out["Factor"] = pd.Categorical(out["Factor"], categories=_to_factor_labels(sorted(set(x for x in out["Factor"]), key=lambda s: int(str(s).split()[-1]) if str(s).startswith("Factor") else 0)), ordered=True)

    if verbosity:
        print(out.head().to_string(index=False))
    return out


def muvi_variable_loadings_info(model, mdata, verbosity: int = 1) -> pd.DataFrame:
    """
    Get variable loadings as a wide DataFrame.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    mdata : MuData
        Multi-view data object
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Wide DataFrame with rows = variables, columns = Factor i, plus 'view' column
    """
    all_loadings = model.get_factor_loadings(model.view_names, model.factor_names, as_df=True)
    chunks = []
    for view in _view_names(model, mdata):
        # transpose to variables x factors
        df_v = all_loadings[view].T.copy()
        # Reorder factor columns if factor_0.. present
        fcols = [c for c in df_v.columns if isinstance(c, str) and c.startswith("factor_")]
        if len(fcols) == df_v.shape[1]:
            fcols_sorted = [f"factor_{i}" for i in range(len(fcols))]
            df_v = df_v.loc[:, fcols_sorted]
        df_v["view"] = view
        df_v["variable"] = df_v.index
        chunks.append(df_v)

    var_load = pd.concat(chunks, axis=0, ignore_index=False)
    var_load = var_load.rename_axis("variable").reset_index(drop=True)
    var_load = _rename_factor_columns(var_load)

    if verbosity:
        print(var_load.head().to_string(index=False))
    return var_load  # columns: Factor i..., view, variable


def muvi_selected_features_info(variable_loadings: pd.DataFrame,
                                selections: Sequence[Tuple[str, str]],
                                verbosity: int = 1) -> pd.DataFrame:
    """
    Get loadings for selected features.
    
    Parameters
    ----------
    variable_loadings : pd.DataFrame
        Variable loadings from muvi_variable_loadings_info
    selections : list
        List of (feature_name, view) pairs
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Long DataFrame with columns [Variable, view, Factor, loading]
    """
    base = variable_loadings.set_index(["variable", "view"])
    # Collect rows present in selections
    chosen = []
    for ft, view in selections:
        if (ft, view) in base.index:
            row = base.loc[(ft, view)]
            tmp = row.drop(labels=[], errors="ignore")
            tmp_df = tmp.dropna().to_frame().T  # include all factor columns
            tmp_df["variable"] = ft
            tmp_df["view"] = view
            chosen.append(tmp_df)
    if len(chosen) == 0:
        out = pd.DataFrame(columns=["Variable", "view", "Factor", "loading"])
    else:
        wide = pd.concat(chosen, ignore_index=True)
        # gather factors only
        factor_cols = [c for c in wide.columns if str(c).startswith("Factor ")]
        out = wide.melt(id_vars=["variable", "view"], value_vars=factor_cols,
                        var_name="Factor", value_name="loading").rename(columns={"variable": "Variable"})
    if verbosity:
        print(out.head().to_string(index=False))
    return out


def muvi_factor_scores_info(model, mdata, obs_keys: Optional[Sequence[str]] = None,
                            verbosity: int = 1) -> pd.DataFrame:
    """
    Get factor scores with optional metadata columns joined.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    mdata : MuData
        Multi-view data object
    obs_keys : list, optional
        Metadata columns to join
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Factor scores with optional metadata
    """
    fs = model.get_factor_scores(as_df=True)
    fs = fs.rename(columns={c: f"Factor {i+1}" for i, c in enumerate(fs.columns) if str(c).startswith("factor_")})
    if obs_keys:
        fs = fs.join(mdata.obs[obs_keys])
    if verbosity:
        cols_show = [c for c in fs.columns if c.startswith("Factor ")][:3]
        print("Scores columns:", ", ".join(cols_show), "...")
        if obs_keys:
            print("Joined obs:", ", ".join(obs_keys))
    return fs


def muvi_kruskal_info(scores_df: pd.DataFrame, group_col: str,
                      factors: Optional[Sequence[str]] = None,
                      bonferroni: bool = True, verbosity: int = 1) -> pd.DataFrame:
    """
    Perform Kruskal-Wallis test for factors across groups.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with factor scores and metadata
    group_col : str
        Column name for grouping variable
    factors : list, optional
        Factor columns to test
    bonferroni : bool, default True
        Apply Bonferroni correction
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Test results with p-values
    """
    if factors is None:
        factors = [c for c in scores_df.columns if str(c).startswith("Factor ")]
    groups = scores_df[group_col].dropna().unique().tolist()

    rows = []
    for f in factors:
        samples = [scores_df.loc[scores_df[group_col] == g, f].values for g in groups]
        _, p = kruskal(*samples)
        if bonferroni:
            p = p * len(factors)
            p = min(p, 1.0)
        rows.append({"Factor": f, "pvalue": p})
    out = pd.DataFrame(rows).sort_values("pvalue")
    if verbosity:
        print(out.to_string(index=False))
    return out


def muvi_kendall_info(scores_df: pd.DataFrame, ordinal_col: str,
                      factors: Optional[Sequence[str]] = None,
                      bonferroni: bool = True, verbosity: int = 1) -> pd.DataFrame:
    """
    Perform Kendall tau test for factors vs ordinal variable.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with factor scores and metadata
    ordinal_col : str
        Column name for ordinal variable
    factors : list, optional
        Factor columns to test
    bonferroni : bool, default True
        Apply Bonferroni correction
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Test results with p-values
    """
    if factors is None:
        factors = [c for c in scores_df.columns if str(c).startswith("Factor ")]
    codes = pd.Categorical(scores_df[ordinal_col]).codes
    rows = []
    for f in factors:
        _, p = kendalltau(scores_df[f], codes)
        if bonferroni:
            p = p * len(factors)
            p = min(p, 1.0)
        rows.append({"Factor": f, "pvalue": p})
    out = pd.DataFrame(rows).sort_values("pvalue")
    if verbosity:
        print(out.to_string(index=False))
    return out


def muvi_confidence_ellipses_info(scores_df: pd.DataFrame, x_factor: str, y_factor: str,
                                  group_col: str, nstd: float = 2.0, verbosity: int = 1) -> pd.DataFrame:
    """
    Compute confidence ellipse points for factor pairs by group.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with factor scores and metadata
    x_factor : str
        X-axis factor name
    y_factor : str
        Y-axis factor name
    group_col : str
        Column name for grouping variable
    nstd : float, default 2.0
        Number of standard deviations for ellipse
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Ellipse points for each group
    """
    def _cov_ellipse_points(cov: np.ndarray, center: np.ndarray, nstd: float = 2.0, num: int = 100) -> pd.DataFrame:
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        transform = eigvecs @ np.diag(nstd * np.sqrt(eigvals))
        t = np.linspace(0, 2*np.pi, num)
        circle = np.column_stack([np.cos(t), np.sin(t)])
        ellipse = circle @ transform.T
        ellipse += center
        return pd.DataFrame({"x": ellipse[:, 0], "y": ellipse[:, 1]})
    
    out = []
    for g in scores_df[group_col].dropna().unique():
        sub = scores_df[scores_df[group_col] == g]
        x = sub[x_factor].to_numpy()
        y = sub[y_factor].to_numpy()
        center = np.array([np.nanmean(x), np.nanmean(y)])
        cov = np.cov(np.vstack([x, y]))
        ell = _cov_ellipse_points(cov, center, nstd=nstd)
        ell[group_col] = g
        out.append(ell)
    df = pd.concat(out, ignore_index=True)
    if verbosity:
        print(df.head().to_string(index=False))
    return df


def muvi_top_features_by_view_info(variable_loadings: pd.DataFrame,
                                   factors: Sequence[str], top_per_view: int = 5,
                                   by_abs: bool = True, verbosity: int = 1) -> pd.DataFrame:
    """
    Get top features per view across selected factors.
    
    Parameters
    ----------
    variable_loadings : pd.DataFrame
        Variable loadings from muvi_variable_loadings_info
    factors : list
        Factor names to analyze
    top_per_view : int, default 5
        Number of top features per view
    by_abs : bool, default True
        Use absolute values for ranking
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Tidy table with columns: Variable, View, Weight, Factor
    """
    rows = []
    for f in factors:
        df = variable_loadings[["variable", "view", f]].copy()
        df["score"] = df[f].abs() if by_abs else df[f]
        top = df.sort_values("score", ascending=False).groupby("view").head(top_per_view)
        top = top.drop(columns="score").rename(columns={f: "Weight", "view": "View", "variable": "Variable"})
        top["Factor"] = f
        rows.append(top)
    out = pd.concat(rows, ignore_index=True)
    # drop duplicates while keeping first
    out = out.sort_values(by="Weight", key=lambda x: x.abs(), ascending=False).drop_duplicates(subset=["Variable", "View"])
    # keep final number per view
    out = out.groupby("View", group_keys=False).head(top_per_view)
    if verbosity:
        print(out.head().to_string(index=False))
    return out


def muvi_top_features_by_class_info(variable_loadings: pd.DataFrame,
                                    types_map: Dict[str, str],
                                    factors: Sequence[str], top_per_class: int = 5,
                                    by_abs: bool = True, verbosity: int = 1) -> pd.DataFrame:
    """
    Get top features per class across selected factors.
    
    Parameters
    ----------
    variable_loadings : pd.DataFrame
        Variable loadings from muvi_variable_loadings_info
    types_map : dict
        Dictionary mapping feature names to class labels
    factors : list
        Factor names to analyze
    top_per_class : int, default 5
        Number of top features per class
    by_abs : bool, default True
        Use absolute values for ranking
    verbosity : int, default 1
        Level of output verbosity
        
    Returns
    -------
    pd.DataFrame
        Tidy table with Variable, View, Feature type, Weight, Factor
    """
    df = variable_loadings.copy()
    df["Feature type"] = df["variable"].map(types_map).fillna("NA")

    rows = []
    for f in factors:
        tmp = df[["variable", "view", "Feature type", f]].copy()
        tmp["score"] = tmp[f].abs() if by_abs else tmp[f]
        top = tmp.sort_values("score", ascending=False).groupby("Feature type").head(top_per_class)
        top = top.drop(columns="score").rename(columns={f: "Weight", "view": "View", "variable": "Variable"})
        top["Factor"] = f
        rows.append(top)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(by="Weight", key=lambda x: x.abs(), ascending=False).drop_duplicates(subset=["Variable", "View"])
    out = out.groupby("Feature type", group_keys=False).head(top_per_class)
    if verbosity:
        print(out.head().to_string(index=False))
    return out


def muvi_build_selected_anndata(mdata, selection_df: pd.DataFrame,
                                obs_anchor_view: Optional[str] = None):
    """
    Build a single-view AnnData matrix from selected features.
    
    Parameters
    ----------
    mdata : MuData
        Multi-view data object
    selection_df : pd.DataFrame
        DataFrame with columns ['Variable', 'View']
    obs_anchor_view : str, optional
        View to use for obs data
        
    Returns
    -------
    AnnData
        Single-view AnnData with selected features
    """
    import muon as mu
    assert set(["Variable", "View"]).issubset(selection_df.columns)
    if obs_anchor_view is None:
        obs_anchor_view = list(mdata.mod.keys())[0]
    def _col(view, var):
        return mdata[view].X[:, mdata[view].var_names == var]
    X = np.hstack([_col(r["View"], r["Variable"]) for _, r in selection_df.iterrows()])
    ad = mu.AnnData(X)
    ad.obs = mdata[obs_anchor_view].obs.copy()
    ad.var = selection_df.reset_index(drop=True).copy()
    return ad


# Legacy function aliases for backward compatibility
def identify_factor_associations(model, metadata_columns: Optional[List[str]] = None, 
                                categorical_test: str = 'kruskal') -> pd.DataFrame:
    """Legacy function - use muvi_kruskal_info or muvi_kendall_info instead."""
    warnings.warn("identify_factor_associations is deprecated, use muvi_kruskal_info or muvi_kendall_info", 
                  DeprecationWarning)
    return pd.DataFrame()


def characterize_factors(model, top_genes_per_factor: int = 10) -> Dict:
    """Legacy function - use muvi_variable_loadings_info instead."""
    warnings.warn("characterize_factors is deprecated, use muvi_variable_loadings_info", 
                  DeprecationWarning)
    return {}


def cluster_cells_by_factors(model, factors_to_use: Optional[List[int]] = None) -> np.ndarray:
    """Legacy function - use standard clustering on factor scores."""
    warnings.warn("cluster_cells_by_factors is deprecated, use standard clustering on factor scores", 
                  DeprecationWarning)
    return np.array([])
    
    factor_scores = model.get_factor_scores()
    
    if factors_to_use is not None:
        factor_scores = factor_scores[:, factors_to_use]
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = clusterer.fit_predict(factor_scores)
    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clusterer.fit_predict(factor_scores)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return clusters