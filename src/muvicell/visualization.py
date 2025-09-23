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
        Long format DataFrame with Variable, Factor, loading columns
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
        ggplot(df_long, aes(x="Factor", y="Variable", fill="loading"))
        + geom_tile()
        + scale_fill_gradient2(low="#1f77b4", mid="lightgray", high="#c20019", limits=[-1.1, 1.1])
        + theme_classic()
        + theme(axis_text_x=element_text(angle=45, hjust=1), legend_position="bottom")
        + labs(title="Selected features loadings", x="Factor", y="Feature/view", fill="Loading")
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


# Legacy function aliases for backward compatibility
def plot_variance_explained(model, max_factors: Optional[int] = None, by_view: bool = True):
    """Legacy function - use muvi_variance_by_view_plot instead."""
    import warnings
    warnings.warn("plot_variance_explained is deprecated, use muvi_variance_by_view_plot", 
                  DeprecationWarning)
    from .analysis import muvi_variance_by_view_info
    df = muvi_variance_by_view_info(model, verbosity=0)
    return muvi_variance_by_view_plot(df)


def plot_factor_scores(model, factors=(0, 1), color_by=None, size=1.0):
    """Legacy function - use standard scatter plot with factor scores."""
    import warnings
    warnings.warn("plot_factor_scores is deprecated, use standard scatter plot with factor scores", 
                  DeprecationWarning)
    return ggplot()


def plot_factor_loadings(model, view_name, factor=0, top_genes=10):
    """Legacy function - use muvi_plot_top_loadings_heatmap instead."""
    import warnings
    warnings.warn("plot_factor_loadings is deprecated, use muvi_plot_top_loadings_heatmap", 
                  DeprecationWarning)
    return ggplot()


def plot_factor_comparison(model, factors, group_by, plot_type='boxplot'):
    """Legacy function - use muvi_violin_plot instead."""
    import warnings
    warnings.warn("plot_factor_comparison is deprecated, use muvi_violin_plot", 
                  DeprecationWarning)
    return ggplot()
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    factors : Tuple[int, int], default (0, 1)
        Factor indices to plot (x-axis, y-axis)
    color_by : str, optional
        Column in sample metadata to color points by
    size : float, default 1.0
        Point size
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_factor_scores(model, factors=(0, 2), color_by='cell_type')
    >>> p.show()
    """
    # Get factor scores
    if hasattr(model, 'get_factor_scores'):
        factor_scores = model.get_factor_scores()
    else:
        factor_scores = model.mdata.obsm['X_muvi']
    
    if factors[0] >= factor_scores.shape[1] or factors[1] >= factor_scores.shape[1]:
        raise ValueError("Factor indices exceed number of available factors")
    
    # Prepare data
    plot_data = pd.DataFrame({
        f'Factor_{factors[0]}': factor_scores[:, factors[0]],
        f'Factor_{factors[1]}': factor_scores[:, factors[1]]
    })
    
    if color_by is not None:
        # Get metadata from model
        if hasattr(model, 'mdata_original'):
            obs_df = model.mdata_original.obs
        elif hasattr(model, 'mdata'):
            obs_df = model.mdata.obs
        else:
            warnings.warn(f"Cannot access metadata for coloring")
            color_by = None
            obs_df = None
            
        if color_by is not None and obs_df is not None:
            if color_by not in obs_df.columns:
                warnings.warn(f"Column '{color_by}' not found in sample metadata")
                color_by = None
            else:
                plot_data[color_by] = obs_df[color_by].values
    
    # Create plot
    if color_by is not None:
        p = (ggplot(plot_data, aes(x=f'Factor_{factors[0]}', y=f'Factor_{factors[1]}', 
                                  color=color_by)) +
             geom_point(size=size, alpha=0.7) +
             labs(title=f'Factor Scores: Factor {factors[0]} vs Factor {factors[1]}',
                  color=color_by) +
             theme_minimal())
    else:
        p = (ggplot(plot_data, aes(x=f'Factor_{factors[0]}', y=f'Factor_{factors[1]}')) +
             geom_point(size=size, alpha=0.7, color='steelblue') +
             labs(title=f'Factor Scores: Factor {factors[0]} vs Factor {factors[1]}') +
             theme_minimal())
    
    return p


def plot_factor_loadings(
    model,
    view: str,
    factor: int = 0,
    top_genes: int = 20,
    loading_threshold: float = 0.0
) -> ggplot:
    """
    Plot top gene loadings for a specific factor and view.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    view : str
        Name of the view
    factor : int, default 0
        Factor index
    top_genes : int, default 20
        Number of top genes to show
    loading_threshold : float, default 0.0
        Minimum absolute loading threshold
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_factor_loadings(model, 'view1', factor=1, top_genes=15)
    >>> p.show()
    """
    # Get factor loadings
    if hasattr(model, 'get_factor_loadings'):
        loadings_dict = model.get_factor_loadings()
        if view not in loadings_dict:
            raise ValueError(f"View '{view}' not found")
        loadings = loadings_dict[view]
    else:
        # Fall back to mdata access
        if view not in model.mdata.mod:
            raise ValueError(f"View '{view}' not found")
        loadings = model.mdata.mod[view].varm['muvi_loadings']
    
    # Get gene names
    if hasattr(model, 'mdata_original'):
        gene_names = model.mdata_original.mod[view].var_names
    elif hasattr(model, 'mdata'):
        gene_names = model.mdata.mod[view].var_names
    else:
        gene_names = [f"{view}_gene_{i}" for i in range(loadings.shape[0])]
    
    if factor >= loadings.shape[1]:
        raise ValueError(f"Factor {factor} not available")
    
    factor_loadings = loadings[:, factor]
    
    # Filter by threshold and get top genes
    valid_mask = np.abs(factor_loadings) >= loading_threshold
    valid_loadings = factor_loadings[valid_mask]
    valid_genes = gene_names[valid_mask]
    
    # Sort by absolute loading value
    sorted_indices = np.argsort(np.abs(valid_loadings))[::-1][:top_genes]
    
    plot_data = pd.DataFrame({
        'gene': valid_genes[sorted_indices],
        'loading': valid_loadings[sorted_indices],
        'abs_loading': np.abs(valid_loadings[sorted_indices])
    })
    
    # Create plot
    p = (ggplot(plot_data, aes(x='reorder(gene, abs_loading)', y='loading')) +
         geom_col(aes(fill='loading > 0'), alpha=0.8) +
         coord_flip() +
         labs(title=f'Top Gene Loadings: {view} - Factor {factor}',
              x='Genes',
              y='Loading',
              fill='Positive Loading') +
         scale_fill_manual(values=['red', 'blue']) +
         theme_minimal())
    
    return p


def plot_factor_heatmap(
    mdata: mu.MuData,
    view: str,
    factors: Optional[List[int]] = None,
    top_genes_per_factor: int = 10
) -> ggplot:
    """
    Plot heatmap of top gene loadings across factors.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    view : str
        Name of the view
    factors : List[int], optional
        Specific factors to include. If None, uses all factors
    top_genes_per_factor : int, default 10
        Number of top genes per factor
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_factor_heatmap(mdata_muvi, 'rna', factors=[0, 1, 2])
    >>> print(p)
    """
    if view not in mdata.mod:
        raise ValueError(f"View '{view}' not found")
    
    loadings = mdata.mod[view].varm['muvi_loadings']
    gene_names = mdata.mod[view].var_names
    
    if factors is None:
        factors = list(range(loadings.shape[1]))
    
    # Get top genes for each factor
    all_top_genes = set()
    for factor in factors:
        factor_loadings = loadings[:, factor]
        top_indices = np.argsort(np.abs(factor_loadings))[::-1][:top_genes_per_factor]
        all_top_genes.update(gene_names[top_indices])
    
    # Create heatmap data
    heatmap_data = []
    for gene in all_top_genes:
        gene_idx = list(gene_names).index(gene)
        for factor in factors:
            heatmap_data.append({
                'gene': gene,
                'factor': f'Factor_{factor}',
                'loading': loadings[gene_idx, factor]
            })
    
    df = pd.DataFrame(heatmap_data)
    
    # Create heatmap
    p = (ggplot(df, aes(x='factor', y='gene', fill='loading')) +
         geom_tile() +
         scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0) +
         labs(title=f'Factor Loadings Heatmap: {view}',
              x='Factor',
              y='Gene',
              fill='Loading') +
         theme_minimal() +
         theme(axis_text_x=element_text(rotation=45, hjust=1),
               axis_text_y=element_text(size=8)))
    
    return p


def plot_factor_associations(
    mdata: mu.MuData,
    associations_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
    top_n: int = 20
) -> ggplot:
    """
    Plot factor-metadata associations.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    associations_df : pd.DataFrame
        Output from identify_factor_associations()
    p_value_threshold : float, default 0.05
        P-value threshold for significance
    top_n : int, default 20
        Number of top associations to show
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> from MuVIcell.analysis import identify_factor_associations
    >>> assoc = identify_factor_associations(mdata_muvi)
    >>> p = plot_factor_associations(mdata_muvi, assoc)
    >>> print(p)
    """
    # Filter significant associations
    sig_assoc = associations_df[associations_df['p_value'] <= p_value_threshold].copy()
    
    if len(sig_assoc) == 0:
        warnings.warn("No significant associations found")
        return None
    
    # Sort by p-value and take top N
    sig_assoc = sig_assoc.sort_values('p_value').head(top_n)
    
    # Add -log10 p-value for plotting
    sig_assoc['neg_log10_p'] = -np.log10(sig_assoc['p_value'])
    
    # Create plot
    p = (ggplot(sig_assoc, aes(x='reorder(metadata, neg_log10_p)', 
                              y='neg_log10_p', 
                              fill='factor')) +
         geom_col() +
         coord_flip() +
         geom_hline(yintercept=-np.log10(p_value_threshold), 
                    linetype='dashed', color='red') +
         labs(title='Factor-Metadata Associations',
              x='Metadata',
              y='-log10(p-value)',
              fill='Factor') +
         theme_minimal())
    
    return p


def plot_cell_clusters(
    mdata: mu.MuData,
    cluster_labels: np.ndarray,
    factors: Tuple[int, int] = (0, 1),
    cluster_column: str = 'factor_clusters'
) -> ggplot:
    """
    Plot cell clusters in factor space.
    
    Parameters
    ----------
    mdata : mu.MuData
        Muon data object with MuVI results
    cluster_labels : np.ndarray
        Cluster labels for each cell
    factors : Tuple[int, int], default (0, 1)
        Factor indices to plot
    cluster_column : str, default 'factor_clusters'
        Name for cluster column
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> from MuVIcell.analysis import cluster_cells_by_factors
    >>> clusters = cluster_cells_by_factors(mdata_muvi, n_clusters=5)
    >>> p = plot_cell_clusters(mdata_muvi, clusters, factors=(0, 1))
    >>> print(p)
    """
    factor_scores = mdata.obsm['X_muvi']
    
    plot_data = pd.DataFrame({
        f'Factor_{factors[0]}': factor_scores[:, factors[0]],
        f'Factor_{factors[1]}': factor_scores[:, factors[1]],
        cluster_column: [f'Cluster_{i}' for i in cluster_labels]
    })
    
    p = (ggplot(plot_data, aes(x=f'Factor_{factors[0]}', y=f'Factor_{factors[1]}', 
                              color=cluster_column)) +
         geom_point(alpha=0.7, size=1.5) +
         labs(title=f'Cell Clusters in Factor Space (Factors {factors[0]} & {factors[1]})',
              color='Cluster') +
         theme_minimal())
    
    return p


def plot_factor_comparison(
    model,
    factors: List[int],
    group_by: str,
    plot_type: str = 'boxplot'
) -> ggplot:
    """
    Compare factor activity across sample groups.
    
    Parameters
    ----------
    model : MuVI model
        Fitted MuVI model object
    factors : List[int]
        Factor indices to compare
    group_by : str
        Column in sample metadata to group by
    plot_type : str, default 'boxplot'
        Type of plot ('boxplot', 'violin')
        
    Returns
    -------
    ggplot
        Plotnine plot object
        
    Examples
    --------
    >>> p = plot_factor_comparison(model, [0, 1, 2], 'cell_type')
    >>> p.show()
    """
    # Get factor scores
    if hasattr(model, 'get_factor_scores'):
        factor_scores = model.get_factor_scores()
    else:
        factor_scores = model.mdata.obsm['X_muvi']
    
    # Get metadata
    if hasattr(model, 'mdata_original'):
        obs_df = model.mdata_original.obs
    elif hasattr(model, 'mdata'):
        obs_df = model.mdata.obs
    else:
        raise ValueError("Cannot access sample metadata")
    
    if group_by not in obs_df.columns:
        raise ValueError(f"Column '{group_by}' not found in sample metadata")
    
    # Prepare data
    plot_data = []
    for factor_idx in factors:
        for i, score in enumerate(factor_scores[:, factor_idx]):
            plot_data.append({
                'factor': f'Factor_{factor_idx}',
                'score': score,
                'group': obs_df[group_by].iloc[i]
            })
    
    df = pd.DataFrame(plot_data)
    
    # Remove NA groups
    df = df.dropna(subset=['group'])
    
    # Create plot
    if plot_type == 'boxplot':
        p = (ggplot(df, aes(x='group', y='score', fill='group')) +
             geom_boxplot(alpha=0.7) +
             facet_wrap('~factor', scales='free_y') +
             labs(title=f'Factor Activity by {group_by}',
                  x=group_by,
                  y='Factor Score') +
             theme_minimal() +
             theme(axis_text_x=element_text(rotation=45, hjust=1),
                   legend_position='none'))
    elif plot_type == 'violin':
        p = (ggplot(df, aes(x='group', y='score', fill='group')) +
             geom_violin(alpha=0.7) +
             facet_wrap('~factor', scales='free_y') +
             labs(title=f'Factor Activity by {group_by}',
                  x=group_by,
                  y='Factor Score') +
             theme_minimal() +
             theme(axis_text_x=element_text(rotation=45, hjust=1),
                   legend_position='none'))
    else:
        raise ValueError("plot_type must be 'boxplot' or 'violin'")
    
    return p