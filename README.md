# MuVIcell

**From cell-type stratified features to multicellular coordinated programs**

MuVIcell is a Python package for multi-view integration and analysis of single-cell data using MuVI (Multi-View Integration). It provides a streamlined workflow for analyzing multi-modal single-cell datasets, identifying latent factors that capture coordinated programs across different molecular layers.

## Features

- 🧬 **Multi-view integration** using MuVI for single-cell data
- 📊 **Comprehensive preprocessing** pipeline with sensible defaults
- 🔍 **Factor analysis and interpretation** tools
- 📈 **Rich visualization** capabilities using plotnine
- 🧪 **Synthetic data generation** for testing and development
- ✅ **Extensive testing** to ensure reliability
- 📚 **Complete documentation** and tutorials

## Installation

### Requirements

**Important**: MuVI requires Python 3.9-3.10. Python 3.11+ is not currently supported by MuVI.

For the full MuVI functionality:
```bash
# Create a Python 3.10 environment
conda create -n muvicell python=3.10
conda activate muvicell

# Clone and install
git clone https://github.com/HartmannLab/MuVIcell.git
cd MuVIcell
pip install -e .[muvi]
```

For development without MuVI (Python 3.11+ compatible):
```bash
# Clone the repository  
git clone https://github.com/HartmannLab/MuVIcell.git
cd MuVIcell

# Install without MuVI (will use mock implementation)
pip install -e .
```

### Dependencies

Core dependencies:
- `muon` - Multi-modal omics data handling
- `plotnine` - Grammar of graphics visualization
- `scanpy` - Single-cell analysis
- `pandas`, `numpy`, `scipy` - Data manipulation and analysis

Optional dependencies:
- `muvi` - Multi-view integration (requires Python 3.9-3.10)

## Quick Start

```python
import numpy as np
from muvicell import synthetic, preprocessing, muvi_runner, analysis, visualization

# 1. Generate synthetic multi-view data
mdata = synthetic.generate_synthetic_data(
    n_cells=200,
    view_configs={
        'rna': {'n_vars': 100, 'sparsity': 0.3},
        'protein': {'n_vars': 50, 'sparsity': 0.2}
    }
)

# 2. Preprocess for MuVI analysis
mdata_processed = preprocessing.preprocess_for_muvi(mdata)

# 3. Run MuVI
mdata_muvi = muvi_runner.run_muvi(mdata_processed, n_factors=10)

# 4. Analyze factors
factor_genes = analysis.characterize_factors(mdata_muvi)
associations = analysis.identify_factor_associations(mdata_muvi)

# 5. Visualize results
p1 = visualization.plot_variance_explained(mdata_muvi)
p2 = visualization.plot_factor_scores(mdata_muvi, color_by='cell_type')
```

## Tutorial

See the [MuVIcell Tutorial](notebooks/MuVIcell_Tutorial.ipynb) Jupyter notebook for a comprehensive walkthrough of the package functionality.

## Package Structure

```
muvicell/
├── src/muvicell/
│   ├── data.py              # Data loading and validation
│   ├── preprocessing.py     # Data preprocessing utilities
│   ├── muvi_runner.py      # MuVI wrapper functions
│   ├── analysis.py         # Factor analysis and interpretation
│   ├── visualization.py    # Plotting functions
│   └── synthetic.py        # Synthetic data generation
├── tests/                  # Comprehensive test suite
├── notebooks/             # Tutorial notebooks
└── docs/                  # Documentation
```

## Core Modules

### 1. Data Handling (`data.py`)
- Load and save muon data objects
- Validate data for MuVI analysis
- Get view information and statistics

### 2. Preprocessing (`preprocessing.py`)
- Normalize multi-view data
- Filter cells and genes
- Find highly variable genes
- Complete preprocessing pipeline

### 3. MuVI Runner (`muvi_runner.py`)
- Set up and run MuVI models
- Extract factor scores and loadings
- Calculate variance explained
- Select top factors

### 4. Analysis (`analysis.py`)
- Characterize factors by top genes
- Identify factor-metadata associations
- Cluster cells based on factors
- Calculate factor correlations

### 5. Visualization (`visualization.py`)
- Plot variance explained
- Visualize factor scores and loadings
- Show factor-metadata associations
- Create factor heatmaps

### 6. Synthetic Data (`synthetic.py`)
- Generate realistic multi-view datasets
- Add latent factor structure
- Simulate batch effects and missing data

## Workflow Overview

MuVIcell follows a structured workflow:

1. **Data Loading/Generation**
   - Load real multi-view data or generate synthetic data
   - Validate data structure for MuVI compatibility

2. **Preprocessing**
   - Normalize expression data
   - Filter low-quality cells and genes
   - Identify highly variable genes

3. **MuVI Analysis**
   - Run multi-view integration
   - Identify latent factors across views
   - Extract factor scores and loadings

4. **Factor Interpretation**
   - Characterize factors by contributing genes
   - Identify associations with cell metadata
   - Cluster cells based on factor activity

5. **Visualization**
   - Create publication-ready plots
   - Visualize factor structure and interpretation

## Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=MuVIcell --cov-report=html
```

## Development

### Setting up a development environment

```bash
# Clone and install in development mode
git clone https://github.com/HartmannLab/MuVIcell.git
cd MuVIcell

# Install with development dependencies
uv sync --extra dev

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Ensure tests pass (`pytest tests/`)
5. Format code (`black .` and `isort .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Citation

If you use MuVIcell in your research, please cite:

```bibtex
@software{muvicell2024,
  title={MuVIcell: From cell-type stratified features to multicellular coordinated programs},
  author={HartmannLab},
  year={2024},
  url={https://github.com/HartmannLab/MuVIcell}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MuVI](https://github.com/gtca/muvi) for multi-view integration methodology
- [muon](https://github.com/scverse/muon) for multi-modal data handling
- [scanpy](https://github.com/scverse/scanpy) for single-cell analysis tools
- [plotnine](https://github.com/has2k1/plotnine) for grammar of graphics visualization

## Support

For questions, issues, or contributions:
- 📧 Contact: [HartmannLab](mailto:info@hartmannlab.org)
- 🐛 Issues: [GitHub Issues](https://github.com/HartmannLab/MuVIcell/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/HartmannLab/MuVIcell/discussions)
