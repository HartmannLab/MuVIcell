"""
Comprehensive test of the complete notebook workflow.

This test validates that every step in the MuVIcell_Tutorial.ipynb notebook
works correctly and produces expected results.
"""

import numpy as np
import pandas as pd
import muon as mu

from muvicell import synthetic, preprocessing, analysis, visualization


def test_complete_notebook_workflow():
    """Test the complete workflow from the notebook."""
    
    # Step 1: Generate synthetic data
    mdata = synthetic.generate_synthetic_data(
        n_samples=200,
        n_true_factors=3,
        view_configs={
            'view1': {'n_vars': 5, 'sparsity': 0.3},
            'view2': {'n_vars': 10, 'sparsity': 0.4},
            'view3': {'n_vars': 15, 'sparsity': 0.5}
        }
    )
    
    # Validate data structure
    assert mdata.n_obs == 200
    assert len(mdata.mod) == 3
    assert mdata.mod['view1'].n_vars == 5
    assert mdata.mod['view2'].n_vars == 10
    assert mdata.mod['view3'].n_vars == 15
    
    # Step 2: Add latent structure
    mdata_structured = synthetic.add_latent_structure(
        mdata, 
        n_latent_factors=3
    )
    
    # Check metadata was added
    assert 'cell_type' in mdata_structured.obs.columns
    assert 'condition' in mdata_structured.obs.columns
    assert len(mdata_structured.obs['cell_type'].unique()) >= 2
    
    # Step 3: Preprocess
    mdata_processed = preprocessing.preprocess_for_muvi(
        mdata_structured,
        filter_cells=False,
        filter_genes=False,
        normalize=True,
        find_hvg=False,
        subset_hvg=False
    )
    
    assert mdata_processed.shape == (200, 30)
    
    # Step 4: Run MuVI (mock version)
    from muvicell.muvi_runner import _create_mock_muvi_model
    model = _create_mock_muvi_model(mdata_processed, n_factors=3)
    model.fit()
    
    # Validate model results
    assert hasattr(model, 'get_factor_scores')
    assert hasattr(model, 'get_factor_loadings')
    
    factor_scores = model.get_factor_scores()
    assert factor_scores.shape == (200, 3)
    
    factor_loadings = model.get_factor_loadings()
    assert len(factor_loadings) == 3  # 3 views
    assert 'view1' in factor_loadings
    assert factor_loadings['view1'].shape == (5, 3)  # 5 genes, 3 factors
    
    # Step 5: Characterize factors
    factor_genes = analysis.characterize_factors(
        model, 
        top_genes_per_factor=3
    )
    
    assert isinstance(factor_genes, dict)
    assert len(factor_genes) == 3  # 3 views
    for view_name, view_genes in factor_genes.items():
        assert isinstance(view_genes, pd.DataFrame)
        if len(view_genes) > 0:
            assert 'factor' in view_genes.columns
            assert 'gene' in view_genes.columns
            assert 'loading' in view_genes.columns
    
    # Step 6: Factor associations
    associations = analysis.identify_factor_associations(
        model,
        categorical_test='kruskal'
    )
    
    assert isinstance(associations, pd.DataFrame)
    if len(associations) > 0:
        assert 'factor' in associations.columns
        assert 'metadata' in associations.columns
        assert 'p_value' in associations.columns
    
    # Step 7: Clustering
    clusters = analysis.cluster_cells_by_factors(
        model,
        factors_to_use=None,
        n_clusters=3
    )
    
    assert len(clusters) == 200
    assert len(np.unique(clusters)) <= 3  # At most 3 clusters
    
    # Step 8: Visualizations
    # Test each visualization function
    p1 = visualization.plot_variance_explained(model, max_factors=3)
    assert p1 is not None
    
    p2 = visualization.plot_factor_scores(model, factors=(0, 1), color_by='cell_type')
    assert p2 is not None
    
    p3 = visualization.plot_factor_loadings(model, 'view1', factor=0, top_genes=5)
    assert p3 is not None
    
    p4 = visualization.plot_factor_comparison(
        model,
        factors=[0, 1, 2],
        group_by='cell_type',
        plot_type='boxplot'
    )
    assert p4 is not None
    
    # Step 9: Summary analysis
    factor_corr = analysis.calculate_factor_correlations(model)
    assert factor_corr.shape == (3, 3)
    assert isinstance(factor_corr, pd.DataFrame)
    
    # Check factor score statistics
    assert np.all(np.isfinite(factor_scores))
    assert factor_scores.std() > 0  # Some variation in scores
    
    print("✅ Complete notebook workflow test passed!")
    print(f"   - Generated {mdata.n_obs} samples with {sum(v.n_vars for v in mdata.mod.values())} features")
    print(f"   - Identified {factor_scores.shape[1]} latent factors")
    print(f"   - Created {len(factor_genes)} factor characterizations")
    print(f"   - Generated 4 visualization plots")
    print(f"   - Clustered samples into {len(np.unique(clusters))} groups")


if __name__ == "__main__":
    test_complete_notebook_workflow()