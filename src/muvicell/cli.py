"""
Command-line interface for MuVIcell.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .core import MuVIcellAnalyzer
from .utils import load_data, validate_data, generate_sample_data


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_analyze(args) -> None:
    """Run analysis on provided data."""
    analyzer = MuVIcellAnalyzer()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = analyzer.load_data(args.input)
    
    # Validate data
    if args.validate:
        print("Validating data...")
        validation = validate_data(data)
        if not validation['validation_passed']:
            print("Data validation failed:")
            for issue in validation['issues']:
                print(f"  - {issue}")
            return
        print("Data validation passed!")
    
    # Preprocess data
    if args.preprocess:
        print("Preprocessing data...")
        analyzer.preprocess_data(
            normalize=args.normalize,
            remove_outliers=args.remove_outliers
        )
    
    # Analyze cell features
    if args.cell_type_col and args.feature_cols:
        print("Analyzing cell features...")
        feature_cols = args.feature_cols.split(',')
        results = analyzer.analyze_cell_features(
            cell_type_col=args.cell_type_col,
            feature_cols=feature_cols,
            method=args.method
        )
        
        print("\nAnalysis Results:")
        for cell_type, features in results.items():
            print(f"  {cell_type}: {features}")
    
    # Print summary
    summary = analyzer.get_summary()
    print(f"\nAnalysis Summary:")
    print(f"  Data shape: {summary['data_shape']}")
    print(f"  Available results: {summary['results_available']}")


def cmd_generate_sample(args) -> None:
    """Generate sample data for testing."""
    print(f"Generating sample data...")
    data = generate_sample_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_cell_types=args.n_cell_types,
        random_state=args.random_state
    )
    
    output_path = Path(args.output)
    data.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")


def cmd_validate(args) -> None:
    """Validate data file."""
    print(f"Validating data from {args.input}...")
    data = load_data(args.input)
    
    required_cols = args.required_columns.split(',') if args.required_columns else None
    validation = validate_data(data, required_columns=required_cols)
    
    print(f"Data shape: {validation['shape']}")
    print(f"Columns: {validation['columns']}")
    print(f"Duplicate rows: {validation['duplicate_rows']}")
    
    if validation['missing_values']:
        print("Missing values:")
        for col, count in validation['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}")
    
    if validation['validation_passed']:
        print("✓ Data validation passed!")
    else:
        print("✗ Data validation failed:")
        for issue in validation['issues']:
            print(f"  - {issue}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MuVIcell: From cell-type stratified features to multicellular coordinated programs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze data')
    analyze_parser.add_argument('input', help='Input data file path')
    analyze_parser.add_argument('--validate', action='store_true', help='Validate data before analysis')
    analyze_parser.add_argument('--preprocess', action='store_true', help='Preprocess data')
    analyze_parser.add_argument('--normalize', action='store_true', help='Normalize features')
    analyze_parser.add_argument('--remove-outliers', action='store_true', help='Remove outliers')
    analyze_parser.add_argument('--cell-type-col', help='Cell type column name')
    analyze_parser.add_argument('--feature-cols', help='Comma-separated feature column names')
    analyze_parser.add_argument('--method', default='mean', choices=['mean', 'median', 'std'], 
                               help='Analysis method')
    
    # Generate sample command
    sample_parser = subparsers.add_parser('generate-sample', help='Generate sample data')
    sample_parser.add_argument('output', help='Output file path')
    sample_parser.add_argument('--n-samples', type=int, default=100, help='Number of samples')
    sample_parser.add_argument('--n-features', type=int, default=5, help='Number of features')
    sample_parser.add_argument('--n-cell-types', type=int, default=3, help='Number of cell types')
    sample_parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data file')
    validate_parser.add_argument('input', help='Input data file path')
    validate_parser.add_argument('--required-columns', help='Comma-separated required column names')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'generate-sample':
            cmd_generate_sample(args)
        elif args.command == 'validate':
            cmd_validate(args)
    except Exception as e:
        logging.error(f"Error executing command: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()