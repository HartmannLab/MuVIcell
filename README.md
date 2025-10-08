# MuVIcell
**From cell-type stratified features to multicellular coordinated programs**

[![DOI](https://zenodo.org/badge/1060213785.svg)](https://doi.org/10.5281/zenodo.17186801)

MuVIcell is a comprehensive Python package for multi-view integration and analysis of sample-aggregated single-cell data using MuVI (Multi-View Integration). The package enables researchers to identify latent factors that capture coordinated programs across different molecular layers (e.g., RNA, protein, chromatin accessibility) in multicellular systems.

## Overview

MuVIcell provides a streamlined workflow for analyzing multi-modal datasets where:
- Each **row** represents a **sample** (not individual cells)
- **Views** contain **cell type aggregated data per sample**
- The goal is to identify latent factors capturing coordinated multicellular programs

## Key Features

- **Complete MuVI Integration**: Uses the exact same API as the original MOFACell analysis
- **Synthetic Data Generation**: Create realistic multi-view datasets for testing and development
- **Comprehensive Analysis**: Factor characterization, variance analysis, and statistical testing
- **Rich Visualizations**: Publication-ready plots using plotnine
- **Robust Testing**: Extensive test suite ensuring reliability

## Installation

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.6.0
- MuVI (Multi-View Integration)

### Installation Steps

**With uv (recommend):**
```bash
uv sync
```

**With conda:**
```bash
conda create -n muvicell python=3.11
conda activate muvicell
git clone https://github.com/HartmannLab/MuVIcell.git
pip install -e MuVIcell
```

**With pip (global or venv, not recommended):**
```bash
git clone https://github.com/HartmannLab/MuVIcell.git
pip install -e MuVIcell
```

## Quick Start

### Basic Workflow

```python
import muvi
import muvi.tl
from muvicell import synthetic, preprocessing, analysis, visualization

# 1. Generate synthetic multi-view data
mdata = synthetic.generate_synthetic_data(
    n_samples=200,
    n_true_factors=3,
    view_configs={
        'view1': {'n_vars': 5, 'sparsity': 0.3},
        'view2': {'n_vars': 10, 'sparsity': 0.4},
        'view3': {'n_vars': 15, 'sparsity': 0.5}
    }
)

# 2. Add latent structure
mdata_structured = synthetic.add_latent_structure(mdata, n_latent_factors=3)

# 3. Preprocess for MuVI analysis
mdata_processed = preprocessing.preprocess_for_muvi(mdata_structured)

# 4. Run MuVI using exact same API as original analysis
model = muvi.tl.from_mdata(mdata_processed, n_factors=3, nmf=False)
model.fit()

# 5. Analyze results
reconstruction_stats = analysis.muvi_reconstruction_info(model, mdata_processed)
variance_df = analysis.muvi_variance_by_view_info(model)
factor_scores = analysis.muvi_factor_scores_info(model, mdata_processed, obs_keys=['cell_type', 'condition'])

# 6. Create visualizations
p1 = visualization.muvi_variance_by_view_plot(variance_df)
p1.show()

p2 = visualization.muvi_violin_plot(factor_scores, "Factor 1", "cell_type")
p2.show()
```

## Package Structure

The package includes six core modules:

### `synthetic.py`
Generate realistic synthetic multi-view data with configurable latent structure:
- `generate_synthetic_data()`: Create multi-view datasets
- `add_latent_structure()`: Add realistic factor structure

### `preprocessing.py`
Complete preprocessing pipeline for MuVI analysis:
- `preprocess_for_muvi()`: All-in-one preprocessing function
- `normalize_views()`: View-specific normalization
- `find_highly_variable_genes()`: HVG detection across views

### `analysis.py`
Comprehensive factor analysis and interpretation:
- `muvi_reconstruction_info()`: Reconstruction quality assessment
- `muvi_variance_by_view_info()`: Variance explained analysis
- `muvi_variable_loadings_info()`: Feature loadings extraction
- `muvi_factor_scores_info()`: Factor scores with metadata
- `muvi_kruskal_info()`: Statistical testing for categorical variables
- `muvi_kendall_info()`: Statistical testing for ordinal variables

### `visualization.py`
Publication-ready plotting functions using plotnine:
- `muvi_variance_by_view_plot()`: Variance heatmaps with marginals
- `muvi_reconstruction_plot()`: Reconstruction quality plots
- `muvi_plot_top_loadings_heatmap()`: Feature loading heatmaps
- `muvi_violin_plot()`: Factor distribution plots
- `muvi_confidence_ellipses_plot()`: Group confidence ellipses

### `data.py`
Data loading and validation utilities:
- `validate_for_muvi()`: Ensure data compatibility
- `get_view_info()`: Extract view metadata

## Tutorial and Examples

### Jupyter Notebook Tutorial
The package includes a comprehensive tutorial notebook (`notebooks/MuVIcell_Tutorial.ipynb`) that demonstrates:
- Complete workflow from synthetic data generation to final analysis
- All major analysis and visualization functions
- Statistical testing and interpretation
- Best practices for real data analysis

### Real Data Analysis Example
For an example of the workflow applied to real data, see the original MOFACell analysis:
[MIBI Analysis Hamburg CRC TMA 2024](https://github.com/HartmannLab/MIBI-Analysis_Hamburg_CRC_TMA_2024/blob/main/notebooks/multicellular/MOFACell.ipynb)

### Starting from cell-level measurements
The examplke notebook starts from a `muon` object (multiview `anndata`). If you start your analysis from cell-level measurements in a tabular format, you will need to aggregate (e.g. pseudobulk) data per cell type. An example can be found here: https://github.com/HartmannLab/MIBI-Analysis_Hamburg_CRC_TMA_2024/blob/main/notebooks/multicellular/PrepareMuData.ipynb

## API Reference

### Core Analysis Functions

#### Reconstruction Analysis
```python
reconstruction_stats = analysis.muvi_reconstruction_info(model, mdata)
plot = visualization.muvi_reconstruction_plot(reconstruction_stats['by_view'])
```

#### Variance Analysis
```python
variance_df = analysis.muvi_variance_by_view_info(model)
plot = visualization.muvi_variance_by_view_plot(variance_df)
```

#### Factor Characterization
```python
loadings_df = analysis.muvi_variable_loadings_info(model, mdata)
plot = visualization.muvi_plot_top_loadings_heatmap(loadings_df, factor="Factor 1")
```

#### Statistical Testing
```python
scores_df = analysis.muvi_factor_scores_info(model, mdata, obs_keys=['group'])
kruskal_results = analysis.muvi_kruskal_info(scores_df, 'group')
violin_plot = visualization.muvi_violin_plot(scores_df, "Factor 1", 'group')
```

## Data Format

MuVIcell works with MuData objects where:
- **Observations (rows)**: Samples (e.g., patients, conditions, time points)
- **Views**: Different molecular layers or data types
- **Features**: Cell type aggregated measurements per view
- **Metadata**: Sample-level annotations (stored in `mdata.obs`)

## Testing

Run the test suite to ensure everything works correctly:

```bash
pytest tests/
```

The test suite includes:
- Unit tests for all modules
- Integration tests for complete workflows
- Notebook execution tests

## Documentation

- **API Documentation**: Comprehensive function reference in `docs/API.md`
- **Tutorial Notebook**: Step-by-step workflow guide
- **Examples**: Real data analysis examples

## Acknowledgments

This package builds upon and acknowledges:
- **[LIANA+](https://github.com/saezlab/liana-py)**: For handling and interpreting factors
- **[MOFAcell](https://github.com/saezlab/MOFAcell)**: As conceptual inspiration
- **[MuVI](https://github.com/MLO-lab/MuVI)**: For multi-view integration methodology

## Citation

If you use MuVIcell in your research, please cite our preprint: https://arxiv.org/abs/2510.05083

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions, issues, or feature requests, please:
1. Check the [tutorial notebook](notebooks/MuVIcell_Tutorial.ipynb)
2. Review the [API documentation](docs/API.md)
3. Open an issue on GitHub
