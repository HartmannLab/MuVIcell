"""
Visualization utilities for MuVIcell using plotnine.

This module provides comprehensive functions for visualizing MuVI results including
factor scores, loadings, variance explained, and factor associations.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Sequence
from plotnine import (
    ggplot, aes, geom_col, geom_rect, geom_text, geom_tile, geom_violin, geom_path,
    scale_fill_gradientn, scale_fill_gradient2, scale_fill_manual,
    scale_x_continuous, scale_y_continuous,
    theme_classic, theme, element_text, coord_fixed, coord_flip, coord_equal,
    labs, ggtitle, guides, ggsave
)


def _ggsave_if(p, save_path: Optional[str], width: float = 6, height: float = 4, dpi: int = 300, verbose: bool = False):
    """Save plot if save_path is provided."""
    if save_path:
        ggsave(save_path, plot=p, width=width, height=height, dpi=dpi, verbose=verbose)


def muvi_reconstruction_plot(stats_df: pd.DataFrame,
                             title: str = "Reconstruction R2 by view",
                             save_path: Optional[str] = None,
                             width: float = 6, height: float = 4, dpi: int = 300):
    """
    Bar plot of R2 per view.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with reconstruction statistics
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    width : float, default 6
        Plot width
    height : float, default 4
        Plot height
    dpi : int, default 300
        Plot resolution
        
    Returns
    -------
    ggplot
        Plotnine plot object
    """
    p = (
        ggplot(stats_df, aes(x="view", y="R2"))
        + geom_col()
        + theme_classic()
        + theme(axis_text_x=element_text(angle=45, hjust=1))
        + labs(title=title, x="View", y="R2")
    )
    _ggsave_if(p, save_path, width, height, dpi)
    return p


def muvi_variance_by_view_plot(df: pd.DataFrame,
                               subtitle: Optional[str] = None,
                               save_path: Optional[str] = None,
                               width: float = 6, height: float = 5, dpi: int = 300):
    """
    Heatmap with marginal sums for variance explained by view.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long DataFrame with Factor, View, Variance columns
    subtitle : str, optional
        Plot subtitle
    save_path : str, optional
        Path to save plot
    width : float, default 6
        Plot width
    height : float, default 5
        Plot height
    dpi : int, default 300
        Plot resolution
        
    Returns
    -------
    ggplot
        Plotnine plot object
    """
    # Compute marginals
    row_sums = df.groupby("View", as_index=False)["Variance"].sum()
    row_sums["Factor"] = "Sum"

    col_sums = df.groupby("Factor", as_index=False)["Variance"].sum()
    col_sums["View"] = "Sum"

    # sort views descending by total variance
    sorted_views = row_sums.sort_values("Variance", ascending=False)["View"].tolist()

    # Prepare extended frame with 'Sum' row and col
    factor_levels = list(df["Factor"].cat.categories if isinstance(df["Factor"].dtype, pd.CategoricalDtype) else df["Factor"].unique())
    factor_levels = list(factor_levels) + ["Sum"]
    view_levels = sorted_views + ["Sum"]

    dfm = pd.concat([df, row_sums, col_sums], ignore_index=True)
    dfm["Factor"] = pd.Categorical(dfm["Factor"], categories=factor_levels, ordered=True)
    dfm["View"] = pd.Categorical(dfm["View"], categories=view_levels, ordered=True)

    # Split for plotting
    main = dfm[(dfm["Factor"] != "Sum") & (dfm["View"] != "Sum")].copy()
    rowb = dfm[(dfm["Factor"] == "Sum") & (dfm["View"] != "Sum")].copy()
    colb = dfm[(dfm["View"] == "Sum") & (dfm["Factor"] != "Sum")].copy()

    # Normalize bar lengths
    rowb["bar_length"] = rowb["Variance"] / rowb["Variance"].max()
    colb["bar_length"] = colb["Variance"] / colb["Variance"].max()
    rowb["Variance_label"] = rowb["Variance"].round(2).astype(str)
    colb["Variance_label"] = colb["Variance"].round(2).astype(str)

    # Positions
    fac_no_sum = [x for x in factor_levels if x != "Sum"]
    view_no_sum = [x for x in view_levels if x != "Sum"]
    fpos = {k: i for i, k in enumerate(fac_no_sum)}
    vpos = {k: i for i, k in enumerate(view_no_sum)}

    main["x"] = main["Factor"].map(fpos)
    main["y"] = main["View"].map(vpos)
    colb["x"] = colb["Factor"].map(fpos)
    rowb["y"] = rowb["View"].map(vpos)

    # tile coords
    main["xmin"] = main["x"] - 0.5
    main["xmax"] = main["x"] + 0.5
    main["ymin"] = main["y"] - 0.5
    main["ymax"] = main["y"] + 0.5

    # top bars
    colb["xmin"] = colb["x"] - 0.5
    colb["xmax"] = colb["x"] + 0.5
    colb["ymin"] = len(view_no_sum) - 0.5
    colb["ymax"] = colb["ymin"] + colb["bar_length"]

    # right bars
    rowb["ymin"] = rowb["y"] - 0.5
    rowb["ymax"] = rowb["y"] + 0.5
    rowb["xmin"] = len(fac_no_sum) - 0.5
    rowb["xmax"] = rowb["xmin"] + rowb["bar_length"]

    p = (
        ggplot()
        + geom_rect(main, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="Variance"))
        + geom_rect(colb, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#acabab")
        + geom_text(colb, aes(x="x", y=main["ymax"].max() + 0.2, label="Variance_label"), va="bottom", size=8)
        + geom_rect(rowb, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#acabab")
        + geom_text(rowb, aes(x=main["xmax"].max() + 0.2, y="y", label="Variance_label"), ha="left", size=8)
        + scale_fill_gradientn(colors=["#EFF822", "#CC4977", "#0F0782"])
        + scale_x_continuous(breaks=list(range(len(fac_no_sum))), labels=fac_no_sum)
        + scale_y_continuous(breaks=list(range(len(view_no_sum))), labels=view_no_sum)
        + theme_classic()
        + theme(axis_text_x=element_text(angle=45, hjust=1))
        + labs(title="Variance explained by MuVI factors", subtitle=subtitle, x="Factor", y="View")
    )
    _ggsave_if(p, save_path, width, height, dpi)
    return p


def muvi_featureclass_variance_plot(df: pd.DataFrame,
                                    save_path: Optional[str] = None,
                                    width: float = 5, height: float = 5, dpi: int = 300):
    """
    Heatmap with marginal sums for feature class vs factor.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long DataFrame with Factor, Feature_type, Variance columns
    save_path : str, optional
        Path to save plot
    width : float, default 5
        Plot width
    height : float, default 5
        Plot height
    dpi : int, default 300
        Plot resolution
        
    Returns
    -------
    ggplot
        Plotnine plot object
    """
    # marginals
    row_sums = df.groupby("Feature_type", as_index=False)["Variance"].sum()
    row_sums["Factor"] = "Sum"
    col_sums = df.groupby("Factor", as_index=False)["Variance"].sum()
    col_sums["Feature_type"] = "Sum"

    # order feature types
    sorted_ft = row_sums.sort_values("Variance", ascending=False)["Feature_type"].tolist()

    # categories
    factor_levels = list(pd.unique(df["Factor"])) + ["Sum"]
    ft_levels = sorted_ft + ["Sum"]

    dfx = pd.concat([df, row_sums, col_sums], ignore_index=True)
    dfx["Factor"] = pd.Categorical(dfx["Factor"], categories=factor_levels, ordered=True)
    dfx["Feature_type"] = pd.Categorical(dfx["Feature_type"], categories=ft_levels, ordered=True)

    main = dfx[(dfx["Factor"] != "Sum") & (dfx["Feature_type"] != "Sum")].copy()
    rowb = dfx[(dfx["Factor"] == "Sum") & (dfx["Feature_type"] != "Sum")].copy()
    colb = dfx[(dfx["Feature_type"] == "Sum") & (dfx["Factor"] != "Sum")].copy()

    rowb["bar_length"] = rowb["Variance"] / rowb["Variance"].max()
    colb["bar_length"] = colb["Variance"] / colb["Variance"].max()
    rowb["Variance_label"] = rowb["Variance"].round(2).astype(str)
    colb["Variance_label"] = colb["Variance"].round(2).astype(str)

    fac_no_sum = [x for x in factor_levels if x != "Sum"]
    ft_no_sum = [x for x in ft_levels if x != "Sum"]
    fpos = {k: i for i, k in enumerate(fac_no_sum)}
    ftpos = {k: i for i, k in enumerate(ft_no_sum)}

    main["x"] = main["Factor"].map(fpos)
    main["y"] = main["Feature_type"].map(ftpos)
    colb["x"] = colb["Factor"].map(fpos)
    rowb["y"] = rowb["Feature_type"].map(ftpos)

    main["xmin"] = main["x"] - 0.5
    main["xmax"] = main["x"] + 0.5
    main["ymin"] = main["y"] - 0.5
    main["ymax"] = main["y"] + 0.5

    colb["xmin"] = colb["x"] - 0.5
    colb["xmax"] = colb["x"] + 0.5
    colb["ymin"] = len(ft_no_sum) - 0.5
    colb["ymax"] = colb["ymin"] + colb["bar_length"]

    rowb["ymin"] = rowb["y"] - 0.5
    rowb["ymax"] = rowb["y"] + 0.5
    rowb["xmin"] = len(fac_no_sum) - 0.5
    rowb["xmax"] = rowb["xmin"] + rowb["bar_length"]

    p = (
        ggplot()
        + geom_rect(main, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="Variance"))
        + geom_rect(colb, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#acabab")
        + geom_text(colb, aes(x="x", y=main["ymax"].max() + 0.2, label="Variance_label"), va="bottom", size=8)
        + geom_rect(rowb, aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#acabab")
        + geom_text(rowb, aes(x=main["xmax"].max() + 0.2, y="y", label="Variance_label"), ha="left", size=8)
        + scale_fill_gradientn(colors=["#EFF822", "#CC4977", "#0F0782"])
        + scale_x_continuous(breaks=list(range(len(fac_no_sum))), labels=fac_no_sum)
        + scale_y_continuous(breaks=list(range(len(ft_no_sum))), labels=ft_no_sum)
        + theme_classic()
        + theme(axis_text_x=element_text(angle=45, hjust=1))
        + labs(title="Variance explained by MuVI factors", x="Factor", y="Feature type")
    )
    _ggsave_if(p, save_path, width, height, dpi)
    return p


def muvi_plot_top_loadings_heatmap(variable_loadings: pd.DataFrame,
                                   factor: str = "Factor 1",
                                   top_n: int = 30, by_abs: bool = True,
                                   save_path: Optional[str] = None,
                                   width: float = 5, height: float = 5, dpi: int = 300):
    """
    Tile heatmap of top features across views for a given factor.
    
    Parameters
    ----------
    variable_loadings : pd.DataFrame
        Variable loadings DataFrame
    factor : str, default "Factor 1"
        Factor to plot
    top_n : int, default 30
        Number of top features to show
    by_abs : bool, default True
        Use absolute values for ranking
    save_path : str, optional
        Path to save plot
    width : float, default 5
        Plot width
    height : float, default 5
        Plot height
    dpi : int, default 300
        Plot resolution
        
    Returns
    -------
    ggplot
        Plotnine plot object
    """
    df = variable_loadings.copy()
    if factor not in df.columns:
        raise ValueError(f"Factor column not found: {factor}")
    key = df[factor].abs() if by_abs else df[factor]
    top_vars = df.assign(score=key).sort_values("score", ascending=False).head(top_n)["variable"].tolist()
    plot_df = df[df["variable"].isin(top_vars)][["variable", "view", factor]].copy()
    p = (
        ggplot(plot_df)
        + aes(x="view", y="variable", fill=factor)
        + geom_tile()
        + scale_fill_gradient2(low="#1f77b4", mid="lightgray", high="#c20019", limits=[-1.1, 1.1])
        + theme_classic()
        + theme(axis_text_x=element_text(angle=45, hjust=1))
        + labs(title=factor, x="View", y="Feature", fill="Loading")
        + coord_fixed()
    )
    _ggsave_if(p, save_path, width, height, dpi)
    return p


def muvi_selected_features_plot(df_long: pd.DataFrame,
                                save_path: Optional[str] = None,
                                width: float = 6, height: float = 5, dpi: int = 300):
    """
    Heatmap of selected feature loadings across factors.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long DataFrame with Variable_view, Factor, loading columns
    save_path : str, optional
        Path to save plot
    width : float, default 6
        Plot width
    height : float, default 5
        Plot height
    dpi : int, default 300
        Plot resolution

    Returns
    -------
    ggplot
        Plotnine plot object
    """
    p = (
        ggplot(df_long, aes(x="Factor", y="Variable_view", fill="loading"))
        + geom_tile()
        + scale_fill_gradient2(low="#1f77b4", mid="lightgray", high="#c20019", limits=[-1.1, 1.1])
        + theme_classic()
        + theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="bottom")
        + labs(title="Selected features loadings", x="Factor", y="Feature/View", fill="Loading")
        + coord_fixed()
    )
    _ggsave_if(p, save_path, width, height, dpi)
    return p


def muvi_violin_plot(scores_df: pd.DataFrame, factor: str, group_col: str,
                     palette: Optional[List[str]] = None, pvalue: Optional[float] = None,
                     save_path: Optional[str] = None,
                     width: float = 4.5, height: float = 4.5, dpi: int = 300):
    """
    Violin plot for one factor across categories in group_col.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with factor scores and metadata
    factor : str
        Factor column name
    group_col : str
        Grouping column name
    palette : list, optional
        Color palette
    pvalue : float, optional
        P-value to display
    save_path : str, optional
        Path to save plot
    width : float, default 4.5
        Plot width
    height : float, default 4.5
        Plot height
    dpi : int, default 300
        Plot resolution
        
    Returns
    -------
    ggplot
        Plotnine plot object
    """
    p = (
        ggplot(scores_df, aes(y=factor, x=group_col, fill=group_col))
        + geom_violin(style="right", scale="width", width=1.25)
        + theme_classic()
        + coord_flip()
        + guides(fill=False)
        + labs(title=f"{factor}" + (f" adjusted p = {np.round(pvalue, 5)}" if pvalue is not None else ""), x=group_col, y=factor)
    )
    if palette is not None:
        p = p + scale_fill_manual(values=palette)
    _ggsave_if(p, save_path, width, height, dpi)
    return p


def muvi_confidence_ellipses_plot(scores_df: pd.DataFrame, ellipses_df: pd.DataFrame,
                                  x_factor: str, y_factor: str, group_col: str,
                                  palette: Optional[List[str]] = None,
                                  save_path: Optional[str] = None,
                                  width: float = 4.5, height: float = 4.5, dpi: int = 300):
    """
    Plot confidence ellipses only.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with factor scores and metadata
    ellipses_df : pd.DataFrame
        DataFrame with ellipse coordinates
    x_factor : str
        X-axis factor name
    y_factor : str
        Y-axis factor name
    group_col : str
        Grouping column name
    palette : list, optional
        Color palette
    save_path : str, optional
        Path to save plot
    width : float, default 4.5
        Plot width
    height : float, default 4.5
        Plot height
    dpi : int, default 300
        Plot resolution
        
    Returns
    -------
    ggplot
        Plotnine plot object
    """
    p = (
        ggplot(scores_df, aes(x=x_factor, y=y_factor, color=group_col))
        + geom_path(ellipses_df, aes(x="x", y="y", group=group_col, color=group_col), size = 3)
        + theme_classic()
        + ggtitle("Confidence ellipses by group")
        + coord_equal()
    )
    if palette is not None:
        p = p + scale_fill_manual(values=palette)
    _ggsave_if(p, save_path, width, height, dpi)
    return p