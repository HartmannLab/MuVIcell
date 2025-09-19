# MuVIcell

**From cell-type stratified features to multicellular coordinated programs**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

MuVIcell is a Python package for analyzing multicellular coordination and cell-type specific features from biological data. It provides tools for data preprocessing, feature analysis, visualization, and interpretation of complex cellular datasets.

## Features

- **Cell-type feature analysis**: Analyze and compare features across different cell types
- **Data preprocessing**: Normalization, outlier removal, and data validation
- **Visualization tools**: Create publication-ready plots and interactive visualizations
- **Dimensionality reduction**: PCA, t-SNE support for data exploration
- **Statistical analysis**: Built-in statistical tests for feature significance
- **Command-line interface**: Easy-to-use CLI for common operations
- **Jupyter notebook integration**: Example notebooks for interactive analysis

## Installation

### Using uv (recommended)

```bash
# Install MuVIcell with basic dependencies
uv add muvicell

# Install with development dependencies
uv add muvicell[dev]

# Install with notebook dependencies
uv add muvicell[notebooks]

# Install with all dependencies
uv add muvicell[all]
```

### Using pip

```bash
# Install MuVIcell with basic dependencies
pip install muvicell

# Install with all optional dependencies
pip install muvicell[all]
```

### From source

```bash
git clone https://github.com/HartmannLab/MuVIcell.git
cd MuVIcell
uv sync --all-extras
```

## Quick Start

### Python API

```python
import pandas as pd
from muvicell import MuVIcellAnalyzer, load_data
from muvicell.utils import generate_sample_data

# Generate sample data or load your own
data = generate_sample_data(n_samples=200, n_features=10, n_cell_types=4)

# Initialize analyzer
analyzer = MuVIcellAnalyzer(data=data)

# Preprocess data
processed_data = analyzer.preprocess_data(normalize=True, remove_outliers=True)

# Analyze cell-type specific features
feature_cols = ['feature_1', 'feature_2', 'feature_3']
results = analyzer.analyze_cell_features(
    cell_type_col='cell_type',
    feature_cols=feature_cols,
    method='mean'
)

# Get analysis summary
summary = analyzer.get_summary()
print(summary)
```

### Command Line Interface

```bash
# Generate sample data
muvicell generate-sample sample_data.csv --n-samples 500 --n-features 20

# Validate data
muvicell validate sample_data.csv --required-columns cell_type,feature_1

# Analyze data
muvicell analyze sample_data.csv \
    --cell-type-col cell_type \
    --feature-cols feature_1,feature_2,feature_3 \
    --preprocess \
    --normalize
```

## Example Notebooks

The package includes comprehensive Jupyter notebooks demonstrating various features:

- **`examples/basic_usage.ipynb`**: Introduction to MuVIcell functionality
- **`examples/advanced_features.ipynb`**: Advanced analysis techniques and custom workflows

To run the notebooks:

```bash
# Install notebook dependencies
uv add muvicell[notebooks]

# Start Jupyter
jupyter notebook examples/
```

## Development Environment

### Setting up with uv

```bash
# Clone the repository
git clone https://github.com/HartmannLab/MuVIcell.git
cd MuVIcell

# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run black src/ tests/
uv run isort src/ tests/
uv run flake8 src/ tests/

# Type checking
uv run mypy src/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=muvicell --cov-report=html

# Run specific test file
uv run pytest tests/test_basic.py -v
```

## API Reference

### Core Classes

#### `MuVIcellAnalyzer`

Main analysis class for MuVIcell operations.

```python
analyzer = MuVIcellAnalyzer(data=None)
```

**Methods:**
- `load_data(filepath, **kwargs)`: Load data from various file formats
- `preprocess_data(normalize=True, remove_outliers=False)`: Preprocess the data
- `analyze_cell_features(cell_type_col, feature_cols, method='mean')`: Analyze cell-type features
- `get_summary()`: Get analysis summary

### Utility Functions

#### Data Handling
- `load_data(filepath, **kwargs)`: Load data from files
- `validate_data(data, required_columns=None)`: Validate data integrity
- `generate_sample_data(n_samples, n_features, n_cell_types)`: Generate sample data

#### Visualization
- `visualize_results(data, x_col, y_col, plot_type='scatter')`: Create visualizations
- `plot_correlation_heatmap(corr_matrix)`: Plot correlation heatmaps

#### Analysis
- `calculate_correlation_matrix(data, method='pearson')`: Calculate correlations

## Configuration

MuVIcell uses reasonable defaults for most operations, but you can customize behavior:

### Data Processing

```python
# Custom preprocessing
processed_data = analyzer.preprocess_data(
    normalize=True,           # Z-score normalization
    remove_outliers=True,     # Remove outliers
    outlier_threshold=3.0     # Z-score threshold for outliers
)
```

### Visualization

```python
# Custom plotting
fig = visualize_results(
    data=data,
    x_col='feature1',
    y_col='feature2',
    hue_col='cell_type',
    plot_type='scatter',      # 'scatter', 'box', 'violin', 'bar'
    figsize=(10, 8),
    save_path='plot.png'
)
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `uv run pytest`
5. Submit a pull request

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MuVIcell in your research, please cite:

```bibtex
@software{muvicell,
  author = {HartmannLab},
  title = {MuVIcell: From cell-type stratified features to multicellular coordinated programs},
  url = {https://github.com/HartmannLab/MuVIcell},
  version = {0.1.0},
  year = {2024}
}
```

## Support

- **Documentation**: [GitHub Wiki](https://github.com/HartmannLab/MuVIcell/wiki)
- **Issues**: [GitHub Issues](https://github.com/HartmannLab/MuVIcell/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HartmannLab/MuVIcell/discussions)

## Changelog

### Version 0.1.0 (Initial Release)

- Core analysis functionality
- Data preprocessing and validation
- Basic visualization tools
- Command-line interface
- Example Jupyter notebooks
- Comprehensive test suite
