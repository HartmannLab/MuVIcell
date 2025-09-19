# MuVIcell

**From cell-type stratified features to multicellular coordinated programs**

## API Reference

This document provides a comprehensive reference for the MuVIcell package API.

### Data Module (`MuVIcell.data`)

#### Functions

- `load_muon_data(path)` - Load muon data from file
- `save_muon_data(mdata, path)` - Save muon data to file  
- `get_view_info(mdata)` - Get information about views
- `validate_muon_data(mdata)` - Validate data for MuVI analysis

### Preprocessing Module (`MuVIcell.preprocessing`)

#### Functions

- `normalize_views(mdata, ...)` - Normalize data in each view
- `filter_views(mdata, ...)` - Filter cells and genes
- `find_highly_variable_genes(mdata, ...)` - Find HVGs
- `preprocess_for_muvi(mdata, ...)` - Complete preprocessing pipeline

### MuVI Runner Module (`MuVIcell.muvi_runner`)

#### Functions

- `setup_muvi_model(mdata, ...)` - Set up MuVI model
- `run_muvi(mdata, ...)` - Run MuVI analysis
- `get_factor_scores(mdata)` - Extract factor scores
- `get_factor_loadings(mdata, view)` - Extract factor loadings
- `select_top_factors(mdata, ...)` - Select top factors

### Analysis Module (`MuVIcell.analysis`)

#### Functions

- `characterize_factors(mdata, ...)` - Characterize factors by genes
- `calculate_factor_correlations(mdata)` - Calculate factor correlations
- `identify_factor_associations(mdata, ...)` - Find factor-metadata associations
- `cluster_cells_by_factors(mdata, ...)` - Cluster cells by factors
- `summarize_factor_activity(mdata, ...)` - Summarize factor activity

### Visualization Module (`MuVIcell.visualization`)

#### Functions

- `plot_variance_explained(mdata, ...)` - Plot variance explained
- `plot_factor_scores(mdata, ...)` - Plot factor scores
- `plot_factor_loadings(mdata, ...)` - Plot factor loadings
- `plot_factor_heatmap(mdata, ...)` - Plot factor heatmap
- `plot_factor_associations(mdata, ...)` - Plot factor associations

### Synthetic Module (`MuVIcell.synthetic`)

#### Functions

- `generate_synthetic_data(...)` - Generate synthetic multi-view data
- `add_realistic_structure(mdata, ...)` - Add latent factor structure
- `generate_batch_effects(mdata, ...)` - Add batch effects
- `simulate_missing_data(mdata, ...)` - Simulate missing data

For detailed parameter descriptions and examples, see the function docstrings and tutorial notebook.