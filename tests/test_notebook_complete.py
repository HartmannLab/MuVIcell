"""
Complete test of the notebook workflow to ensure everything works.
This test mirrors the exact notebook steps to validate functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_complete_notebook_workflow():
    """Test the complete notebook workflow step by step."""
    
    # Step 1: Import all modules
    from muvicell import synthetic, preprocessing, analysis, visualization, muvi_runner
    
    # Step 2: Generate synthetic data with 3 true factors
    mdata = synthetic.generate_synthetic_data(
        n_samples=50,  # Smaller for faster testing
        n_true_factors=3,
        view_configs={
            'view1': {'n_vars': 5, 'sparsity': 0.3},
            'view2': {'n_vars': 10, 'sparsity': 0.4},
            'view3': {'n_vars': 15, 'sparsity': 0.5}
        }
    )
    
    # Verify data structure
    assert mdata.shape == (50, 30)  # 50 samples, 30 total features
    assert len(mdata.mod) == 3  # 3 views
    
    # Step 3: Add latent structure
    mdata_structured = synthetic.add_latent_structure(mdata, n_latent_factors=3)
    
    # Verify latent structure added and global metadata exists
    assert hasattr(mdata_structured, 'uns')
    assert 'cell_type' in mdata_structured.obs.columns  # Global metadata added
    
    # Step 4: Preprocess data
    mdata_processed = preprocessing.preprocess_for_muvi(
        mdata_structured, 
        filter_cells=False,  # Don't filter for small test dataset
        filter_genes=False,
        normalize=True,
        find_hvg=False  # Skip HVG for test
    )
    
    # Verify preprocessing
    assert mdata_processed.shape[0] == 50  # Same number of samples
    
    # Step 5: Run MuVI (mock implementation)
    model = muvi_runner.run_muvi(mdata_processed, n_factors=3)
    model.fit()
    
    # Verify model is fitted
    assert hasattr(model, 'fitted')
    assert model.fitted == True
    
    # Step 6: Test analysis functions
    
    # 6a: Factor characterization
    factor_genes = analysis.characterize_factors(model, top_genes_per_factor=3)
    assert isinstance(factor_genes, dict)
    assert len(factor_genes) == len(mdata.mod)  # One per view
    
    # 6b: Factor associations
    associations = analysis.identify_factor_associations(model)
    assert isinstance(associations, pd.DataFrame)
    
    # 6c: Cell clustering  
    clusters = analysis.cluster_cells_by_factors(model, n_clusters=3)
    assert len(clusters) == 50  # One cluster per sample
    assert len(np.unique(clusters)) <= 3  # At most 3 clusters
    
    # Step 7: Test visualization functions
    
    # 7a: Variance explained plot
    p1 = visualization.plot_variance_explained(model)
    assert p1 is not None
    
    # 7b: Factor scores plot  
    p2 = visualization.plot_factor_scores(model, color_by='cell_type')
    assert p2 is not None
    
    # 7c: Factor loadings plot
    p3 = visualization.plot_factor_loadings(model, 'view1', factor=0, top_genes=3)
    assert p3 is not None
    
    # 7d: Factor comparison plot
    p4 = visualization.plot_factor_comparison(model, factors=[0, 1, 2], group_by='cell_type')
    assert p4 is not None
    
    print("✓ All notebook workflow tests passed!")


def test_muvi_api_compatibility():
    """Test that the package works with the MuVI API pattern."""
    
    from muvicell import synthetic, preprocessing
    from muvicell.muvi_runner import _create_mock_muvi_model
    
    # Generate and preprocess data
    mdata = synthetic.generate_synthetic_data(n_samples=20, n_true_factors=3)
    mdata_processed = preprocessing.preprocess_for_muvi(mdata, 
                                                        filter_cells=False, 
                                                        filter_genes=False,
                                                        normalize=True, 
                                                        find_hvg=False)
    
    # Test mock MuVI API compatibility
    model = _create_mock_muvi_model(mdata_processed, n_factors=3)
    
    # Test that model follows MuVI API patterns
    assert hasattr(model, 'fit')
    assert hasattr(model, 'get_factor_scores')
    assert hasattr(model, 'get_factor_loadings')
    assert hasattr(model, 'mdata_original')
    
    # Test that model must be fitted before use
    try:
        model.get_factor_scores()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Model must be fitted first" in str(e)
    
    # Test after fitting
    model.fit()
    scores = model.get_factor_scores()
    loadings = model.get_factor_loadings()
    
    assert scores.shape == (20, 3)  # 20 samples, 3 factors
    assert len(loadings) == 3  # 3 views
    
    print("✓ MuVI API compatibility tests passed!")


if __name__ == "__main__":
    test_complete_notebook_workflow()
    test_muvi_api_compatibility()
    print("🎉 All tests passed! Package is ready for use.")