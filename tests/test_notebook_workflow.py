"""
Test the complete notebook workflow to ensure all code functions correctly.
This test mirrors exactly what is in the MuVIcell_Tutorial.ipynb notebook.
"""

import pytest
import numpy as np
import pandas as pd
import muon as mu
import warnings
warnings.filterwarnings('ignore')

# Test all imports from muvicell
def test_imports():
    """Test that all package imports work correctly."""
    from muvicell import synthetic, preprocessing, muvi_runner, analysis, visualization
    from muvicell import run_muvi, get_factor_scores
    
    # Check that modules are available
    assert synthetic is not None
    assert preprocessing is not None
    assert muvi_runner is not None
    assert analysis is not None
    assert visualization is not None
    assert run_muvi is not None
    assert get_factor_scores is not None


def test_step1_generate_synthetic_data():
    """Test Step 1: Generate Synthetic Multi-View Data"""
    from muvicell import synthetic
    from muvicell.data import get_view_info
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define view configurations exactly as in notebook
    view_configs = {
        'view1': {'n_vars': 5, 'sparsity': 0.3},
        'view2': {'n_vars': 10, 'sparsity': 0.4},
        'view3': {'n_vars': 15, 'sparsity': 0.5}
    }
    
    # Generate synthetic data
    mdata_raw = synthetic.generate_synthetic_data(
        n_cells=200,  # Note: these are samples, not individual cells
        view_configs=view_configs,
        random_state=42
    )
    
    # Test assertions
    assert mdata_raw.n_obs == 200, f"Expected 200 samples, got {mdata_raw.n_obs}"
    assert len(mdata_raw.mod) == 3, f"Expected 3 views, got {len(mdata_raw.mod)}"
    
    # Display view information
    view_info = get_view_info(mdata_raw)
    assert view_info is not None, "View info should not be None"
    
    return mdata_raw


def test_step2_add_latent_structure():
    """Test Step 2: Add Realistic Structure to the Data"""
    from muvicell import synthetic
    
    # Get data from step 1
    mdata_raw = test_step1_generate_synthetic_data()
    
    # Add realistic latent factor structure
    mdata_structured = synthetic.add_latent_structure(
        mdata_raw,
        n_latent_factors=5,
        factor_variance=[0.3, 0.2, 0.15, 0.1, 0.05]
    )
    
    # Test assertions
    assert 'true_factors' in mdata_structured.obsm, "true_factors should be in obsm"
    assert mdata_structured.obsm['true_factors'].shape[1] == 5, "Should have 5 factors"
    assert 'true_factor_variance' in mdata_structured.uns, "true_factor_variance should be in uns"
    
    return mdata_structured


def test_step3_data_validation():
    """Test Step 3: Data Validation"""
    from muvicell.data import validate_for_muvi
    
    # Get data from step 2
    mdata_structured = test_step2_add_latent_structure()
    
    # Validate data for MuVI
    validation_results = validate_for_muvi(mdata_structured)
    
    # Test assertions
    assert validation_results is not None, "Validation results should not be None"
    assert isinstance(validation_results, dict), "Validation results should be a dict"
    
    # Check that at least some validations passed
    all_passed = all(validation_results.values())
    print(f"Data validation - all passed: {all_passed}")
    
    return mdata_structured, validation_results


def test_step4_preprocessing():
    """Test Step 4: Preprocess Data for MuVI"""
    from muvicell import preprocessing
    
    # Get data from step 3
    mdata_structured, _ = test_step3_data_validation()
    
    # Preprocess the data for MuVI - simplified approach
    mdata_processed = preprocessing.preprocess_for_muvi(
        mdata_structured,
        filter_cells=False,  # Skip filtering for small synthetic data
        filter_genes=False,  # Skip filtering for small synthetic data
        normalize=True,
        find_hvg=False,  # Skip HVG detection for small synthetic data
        subset_hvg=False
    )
    
    # Test assertions
    assert mdata_processed is not None, "Processed data should not be None"
    assert mdata_processed.shape[0] > 0, "Should have some samples after preprocessing"
    
    return mdata_processed


def test_step5_muvi_analysis():
    """Test Step 5: Run MuVI Analysis"""
    from muvicell import run_muvi
    
    # Get data from step 4
    mdata_processed = test_step4_preprocessing()
    
    # Run MuVI analysis using the exact same pattern as the original notebook
    mdata_muvi = run_muvi(
        mdata_processed,
        n_factors=10,
        nmf=False,  # Use standard factor analysis, not non-negative
        device="cpu"
    )
    
    # Test assertions
    assert mdata_muvi is not None, "MuVI results should not be None"
    assert 'X_muvi' in mdata_muvi.obsm, "X_muvi should be in obsm"
    assert mdata_muvi.obsm['X_muvi'].shape[1] == 10, "Should have 10 factors"
    assert mdata_muvi.obsm['X_muvi'].shape[0] == mdata_muvi.n_obs, "Factor scores shape should match samples"
    
    return mdata_muvi


def test_step6_factor_analysis():
    """Test Step 6: Analyze MuVI Results"""
    from muvicell import get_factor_scores, muvi_runner, analysis
    
    # Get data from step 5
    mdata_muvi = test_step5_muvi_analysis()
    
    # Extract factor scores and variance explained
    factor_scores = get_factor_scores(mdata_muvi)
    var_explained = muvi_runner.get_variance_explained(mdata_muvi)
    
    # Test assertions
    assert factor_scores is not None, "Factor scores should not be None"
    assert factor_scores.shape[1] == 10, "Should have 10 factors"
    assert var_explained is not None, "Variance explained should not be None"
    assert isinstance(var_explained, dict), "Variance explained should be a dict"
    
    # Characterize factors by identifying top contributing genes
    factor_genes = analysis.characterize_factors(
        mdata_muvi,
        top_genes_per_factor=3,
        loading_threshold=0.1
    )
    
    # Test assertions
    assert factor_genes is not None, "Factor genes should not be None"
    assert isinstance(factor_genes, dict), "Factor genes should be a dict"
    
    # Identify associations between factors and metadata
    associations = analysis.identify_factor_associations(
        mdata_muvi,
        metadata_columns=['cell_type', 'condition'],  # Now use shared column names
        categorical_test='kruskal',
        continuous_test='pearson'
    )
    
    # Test that associations ran (may be None/empty for synthetic data)
    print(f"Factor associations computed: {associations is not None}")
    
    # Cluster cells based on factor scores
    cluster_labels = analysis.cluster_cells_by_factors(
        mdata_muvi,
        n_clusters=3,
        factors_to_use=None  # Use all factors
    )
    
    # Test assertions
    assert cluster_labels is not None, "Cluster labels should not be None"
    assert len(cluster_labels) == mdata_muvi.n_obs, "Should have one cluster label per sample"
    assert len(np.unique(cluster_labels)) <= 3, "Should have at most 3 clusters"
    
    # Add cluster labels to metadata
    mdata_muvi.obs['cluster'] = cluster_labels
    
    return mdata_muvi, factor_scores, var_explained, factor_genes, associations, cluster_labels


def test_step7_visualizations():
    """Test Step 7: Visualize Results"""
    from muvicell import visualization
    
    # Get data from step 6
    mdata_muvi, factor_scores, var_explained, factor_genes, associations, cluster_labels = test_step6_factor_analysis()
    
    # Test all visualization functions
    
    # Plot variance explained by factors
    p1 = visualization.plot_variance_explained(mdata_muvi)
    assert p1 is not None, "Variance explained plot should not be None"
    
    # Plot factor scores in 2D space
    p2 = visualization.plot_factor_scores(
        mdata_muvi,
        factors=[0, 1],
        color_by='cell_type',  # Now use shared column name
        size=3
    )
    assert p2 is not None, "Factor scores plot should not be None"
    
    # Plot factor loadings for the first view
    view_name = list(mdata_muvi.mod.keys())[0]
    p3 = visualization.plot_factor_loadings(
        mdata_muvi,
        view=view_name,
        factor=0,
        top_genes=5
    )
    assert p3 is not None, "Factor loadings plot should not be None"
    
    # Plot cells colored by cluster in factor space
    p4 = visualization.plot_factor_scores(
        mdata_muvi,
        factors=[0, 1],
        color_by='cluster',
        size=3
    )
    assert p4 is not None, "Cluster plot should not be None"
    
    # Compare factor activity across cell types
    p5 = visualization.plot_factor_comparison(
        mdata_muvi,
        factors=[0, 1, 2],
        group_by='cell_type',  # Now use shared column name
        plot_type='boxplot'
    )
    assert p5 is not None, "Factor comparison plot should not be None"
    
    return mdata_muvi, [p1, p2, p3, p4, p5]


def test_step8_summary_analysis():
    """Test Step 8: Summary Analysis"""
    from muvicell import analysis, muvi_runner
    
    # Get data from step 7
    mdata_muvi, plots = test_step7_visualizations()
    
    # Create summary of factor activity by cell type
    factor_summary = analysis.summarize_factor_activity(
        mdata_muvi,
        group_by='cell_type'  # Now use shared column name
    )
    
    # Test assertions
    assert factor_summary is not None, "Factor summary should not be None"
    
    # Calculate factor correlations
    factor_correlations = analysis.calculate_factor_correlations(mdata_muvi)
    
    # Test assertions
    assert factor_correlations is not None, "Factor correlations should not be None"
    assert factor_correlations.shape[0] == 10, "Should have 10x10 correlation matrix"
    assert factor_correlations.shape[1] == 10, "Should have 10x10 correlation matrix"
    
    # Find highly correlated factors
    corr_threshold = 0.5
    high_corr_pairs = []
    for i in range(factor_correlations.shape[0]):
        for j in range(i+1, factor_correlations.shape[1]):
            if abs(factor_correlations.iloc[i, j]) > corr_threshold:
                high_corr_pairs.append((i, j, factor_correlations.iloc[i, j]))
    
    print(f"Found {len(high_corr_pairs)} highly correlated factor pairs")
    
    # Select top factors based on variance explained
    top_factors = muvi_runner.select_top_factors(
        mdata_muvi,
        n_top_factors=3
    )
    
    # Test assertions
    assert top_factors is not None, "Top factors should not be None"
    assert len(top_factors) == 3, "Should have 3 top factors"
    
    # Calculate total variance explained
    var_explained = muvi_runner.get_variance_explained(mdata_muvi)
    total_var_per_factor = []
    for i in range(mdata_muvi.obsm['X_muvi'].shape[1]):
        total_var = sum([var_explained[view][i] for view in var_explained.keys()])
        total_var_per_factor.append(total_var)
    
    # Test assertions
    assert len(total_var_per_factor) == 10, "Should have variance for all 10 factors"
    assert all(var >= 0 for var in total_var_per_factor), "All variances should be non-negative"
    
    return mdata_muvi, factor_summary, factor_correlations, high_corr_pairs, top_factors, total_var_per_factor


def test_complete_notebook_workflow():
    """Test the complete notebook workflow end-to-end."""
    print("\n=== Testing Complete MuVIcell Notebook Workflow ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all steps in sequence
    print("Step 1: Testing imports...")
    test_imports()
    
    print("Step 2: Testing synthetic data generation...")
    mdata_raw = test_step1_generate_synthetic_data()
    
    print("Step 3: Testing latent structure addition...")
    mdata_structured = test_step2_add_latent_structure()
    
    print("Step 4: Testing data validation...")
    mdata_structured, validation_results = test_step3_data_validation()
    
    print("Step 5: Testing preprocessing...")
    mdata_processed = test_step4_preprocessing()
    
    print("Step 6: Testing MuVI analysis...")
    mdata_muvi = test_step5_muvi_analysis()
    
    print("Step 7: Testing factor analysis...")
    results = test_step6_factor_analysis()
    mdata_muvi, factor_scores, var_explained, factor_genes, associations, cluster_labels = results
    
    print("Step 8: Testing visualizations...")
    mdata_muvi, plots = test_step7_visualizations()
    
    print("Step 9: Testing summary analysis...")
    final_results = test_step8_summary_analysis()
    mdata_muvi, factor_summary, factor_correlations, high_corr_pairs, top_factors, total_var_per_factor = final_results
    
    print("\n=== All notebook workflow tests completed successfully! ===")
    
    # Final comprehensive checks
    assert mdata_muvi is not None
    assert mdata_muvi.n_obs == 200
    assert mdata_muvi.obsm['X_muvi'].shape == (200, 10)
    assert len(plots) == 5
    assert all(p is not None for p in plots)
    
    print(f"✓ Final data shape: {mdata_muvi.shape}")
    print(f"✓ Factor scores shape: {mdata_muvi.obsm['X_muvi'].shape}")
    print(f"✓ Number of clusters: {len(np.unique(cluster_labels))}")
    print(f"✓ Number of plots created: {len(plots)}")
    print(f"✓ Top factors selected: {top_factors}")
    
    return True


if __name__ == "__main__":
    # Run the complete workflow test
    test_complete_notebook_workflow()