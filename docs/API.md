# MuVIcell

**From cell-type stratified features to multicellular coordinated programs**

## API Reference

This document provides a comprehensive reference for the MuVIcell package API.

### Data Module (`muvicell.data`)

#### Functions

- `load_muon_data(path)` - Load muon data from file
- `save_muon_data(mdata, path)` - Save muon data to file  

### Preprocessing Module (`muvicell.preprocessing`)

#### Functions

- `normalize_views(mdata, ...)` - Normalize data in each view
- `filter_views(mdata, ...)` - Filter cells and genes
- `find_highly_variable_genes(mdata, ...)` - Find HVGs
- `subset_to_hvg(mdata, ...)` - Subset to HVGs
- `preprocess_for_muvi(mdata, ...)` - Complete preprocessing pipeline

### Analysis Module (`muvicell.analysis`)

#### Functions

- `muvi_reconstruction_info(model, mdata, ...)` - Assess reconstruction performance
- `muvi_variance_by_view_info(model, ...)` - Analyze variance explained
- `muvi_featureclass_variance_info(model, mdata, ...)` - Variance by feature class
- `muvi_variable_loadings_info(model, mdata, ...)` - Extract variable load
- `muvi_selected_features_info(variable_loadings, ...)` - Get feature loadings
- `muvi_factor_scores_info(model, mdata, ...)` - Get factor scores with metadata
- `muvi_kruskal_info(scores_df, ...)` - Kruskal-Wallis test for categorical variables
- `muvi_kendall_info(scores_df, ...)` - Kendall's tau for ordinal variables
- `muvi_confidence_ellipses_info(scores_df, ...)` - Confidence for factor pairs
- `muvi_top_features_by_view_info(variable_loadings, ...)` - Top features by view
- `muvi_top_features_by_class_info(variable_loadings, ...)` - Top features by class
- `muvi_build_selected_anndata(mdata, selection_df, ...)` - Export to AnnData

### Visualization Module (`MuVIcell.visualization`)

#### Functions

- `muvi_reconstruction_plot(stats_df, ...)` - Plot reconstruction performance
- `muvi_variance_by_view_plot(df, ...)` - Plot variance explained by view
- `muvi_featureclass_variance_plot(df, ...)` - Plot variance by feature class
- `muvi_plot_top_loadings_heatmap(variable_loadings, ...)` - Heatmap of top loadings
- `muvi_selected_features_plot(df_long, ...)` - Plot selected feature loadings
- `muvi_violin_plot(scores_df, ...)` - Violin plots of factor scores
- `muvi_confidence_ellipses_plot(scores_df, ellipses_df, ...)` - Plot confidence

### Synthetic Module (`MuVIcell.synthetic`)

#### Functions

- `generate_synthetic_data(...)` - Generate synthetic multi-view data
- `add_latent_structure(mdata, ...)` - Add latent factor structure
- `generate_batch_effects(mdata, ...)` - Add batch effects
- `simulate_missing_data(mdata, ...)` - Simulate missing data

For detailed parameter descriptions and examples, see the function docstrings and tutorial notebook.