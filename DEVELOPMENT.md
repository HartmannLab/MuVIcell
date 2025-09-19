# MuVIcell Development Environment

This document describes how to set up the development environment for MuVIcell using uv.

## Prerequisites

- Python 3.8 or higher
- uv package manager

## Installation

Install uv if you haven't already:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip
pip install uv
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/HartmannLab/MuVIcell.git
cd MuVIcell
```

2. Install all dependencies:
```bash
uv sync --all-extras
```

3. Activate the virtual environment:
```bash
# uv automatically creates and manages the virtual environment
# Use `uv run` to run commands in the environment
```

## Development Commands

### Running Tests
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=muvicell

# Run specific test file
uv run pytest tests/test_basic.py -v
```

### Code Formatting and Linting
```bash
# Format code with black
uv run black src/ tests/

# Sort imports with isort
uv run isort src/ tests/

# Lint with flake8
uv run flake8 src/ tests/

# Type checking with mypy
uv run mypy src/
```

### Running the CLI
```bash
# Generate sample data
uv run muvicell generate-sample data.csv

# Analyze data
uv run muvicell analyze data.csv --cell-type-col cell_type --feature-cols feature_1,feature_2
```

### Running Jupyter Notebooks
```bash
# Start Jupyter Lab
uv run jupyter lab

# Or Jupyter Notebook
uv run jupyter notebook examples/
```

## Package Management

### Adding Dependencies

```bash
# Add a runtime dependency
uv add pandas

# Add a development dependency
uv add pytest --dev

# Add an optional dependency
uv add plotly --optional notebooks
```

### Updating Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update specific package
uv add pandas@latest
```

## Building and Publishing

### Building the Package
```bash
uv build
```

### Installing Local Development Version
```bash
# Install in editable mode
uv pip install -e .
```

## Environment Management

uv automatically creates and manages virtual environments. The environment is stored in `.venv/` directory.

### Manual Environment Control
```bash
# Create environment explicitly
uv venv

# Activate environment (if needed)
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

## IDE Setup

### VS Code
Add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"]
}
```

### PyCharm
1. Go to Settings → Project → Python Interpreter
2. Add Local Interpreter
3. Select "Existing environment"
4. Point to `.venv/bin/python`

## Troubleshooting

### Common Issues

1. **uv not found**: Make sure uv is installed and in your PATH
2. **Permission errors**: Check file permissions or run with appropriate privileges
3. **Dependency conflicts**: Try removing `.venv/` and running `uv sync` again

### Getting Help
- Check the [uv documentation](https://docs.astral.sh/uv/)
- Open an issue on the [MuVIcell repository](https://github.com/HartmannLab/MuVIcell/issues)