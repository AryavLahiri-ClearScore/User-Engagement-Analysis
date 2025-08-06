#!/usr/bin/env python3
"""
Test script for Fixed Advanced Recommender Visualizations
This script tests the visualization functionality added to fixed_advanced_recommender.py
"""

import sys
from pathlib import Path
from fixed_advanced_recommender import FixedAdvancedRecommender, SystemConfig

def test_visualization_functionality():
    """Test all visualization methods in the fixed recommender"""
    print("🧪 TESTING FIXED ADVANCED RECOMMENDER VISUALIZATIONS")
    print("=" * 65)
    
    try:
        # Test 1: Basic initialization
        print("\n1️⃣ Testing Basic Initialization...")
        recommender = FixedAdvancedRecommender('joined_user_table.csv')
        print("✅ Initialization successful")
        
        # Test 2: User feature generation
        print("\n2️⃣ Testing User Feature Generation...")
        user_features = recommender.create_user_features()
        print(f"✅ Generated {len(user_features)} user features")
        print(f"   Columns: {list(user_features.columns)}")
        
        # Test 3: Financial dashboard visualization
        print("\n3️⃣ Testing Financial Dashboard...")
        try:
            recommender.create_financial_visualizations()
            print("✅ Financial dashboard created successfully")
        except Exception as e:
            print(f"❌ Financial dashboard failed: {e}")
            return False
        
        # Test 4: Clustering visualization
        print("\n4️⃣ Testing Clustering Visualization...")
        try:
            features_2d, pca = recommender.visualize_financial_clustering()
            print("✅ Clustering visualization created successfully")
            print(f"   PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")
        except Exception as e:
            print(f"❌ Clustering visualization failed: {e}")
            return False
        
        # Test 5: Correlation analysis
        print("\n5️⃣ Testing Correlation Analysis...")
        try:
            corr_matrix, significant_corrs = recommender.create_engagement_financial_correlation()
            print("✅ Correlation analysis created successfully")
            print(f"   Found {len(significant_corrs)} significant correlations")
        except Exception as e:
            print(f"❌ Correlation analysis failed: {e}")
            return False
        
        # Test 6: Complete analysis workflow
        print("\n6️⃣ Testing Complete Analysis Workflow...")
        try:
            complete_data = recommender.run_complete_analysis()
            print("✅ Complete analysis workflow successful")
            print(f"   Final dataset: {len(complete_data)} users")
        except Exception as e:
            print(f"❌ Complete analysis failed: {e}")
            return False
        
        print("\n🎉 ALL VISUALIZATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_configurations():
    """Test with different configuration settings"""
    print("\n🔧 TESTING CUSTOM CONFIGURATIONS")
    print("=" * 50)
    
    try:
        # Create custom config
        from fixed_advanced_recommender import (
            SystemConfig, VisualizationConfig, FinancialConfig, 
            EngagementConfig, ClusteringConfig, FileConfig
        )
        
        custom_config = SystemConfig(
            visualization=VisualizationConfig(
                dashboard_figsize=(16, 10),
                correlation_figsize=(10, 8),
                clustering_figsize=(12, 10),
                alpha=0.8,
                colormap_segments='tab10'
            ),
            financial=FinancialConfig(
                excellent_threshold=0.85,
                good_threshold=0.65
            ),
            engagement=EngagementConfig(
                high_engagement_threshold=0.6
            ),
            clustering=ClusteringConfig(
                n_clusters=4
            ),
            files=FileConfig(
                output_directory=Path("test_output")
            )
        )
        
        print("✅ Custom configuration created")
        
        # Test with custom config
        custom_recommender = FixedAdvancedRecommender('joined_user_table.csv', config=custom_config)
        print("✅ Custom recommender initialized")
        
        # Run a single visualization test
        custom_recommender.create_financial_visualizations()
        print("✅ Custom configuration visualization test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom configuration test failed: {e}")
        return False

def quick_demo():
    """Quick demonstration of key features"""
    print("\n🚀 QUICK FEATURE DEMONSTRATION")
    print("=" * 40)
    
    try:
        # Initialize
        demo_recommender = FixedAdvancedRecommender('joined_user_table.csv')
        
        # Generate some data
        data = demo_recommender.create_user_features()
        
        # Show basic statistics (the detailed output will be shown automatically)
        print("\n📊 DATA GENERATION COMPLETE:")
        print(f"✅ Generated {len(data)} users with full feature set")
        print(f"✅ {len(data['financial_category'].unique())} financial categories")
        print(f"✅ {len(data['enhanced_segment'].unique())} unique segments")
        print("📊 Detailed distribution summary was printed above ⬆️")
        
        # Show configuration details
        print("\n⚙️ CONFIGURATION DETAILS:")
        config = demo_recommender.config
        print(f"Financial weights sum: {demo_recommender._check_financial_weights():.3f}")
        print(f"Visualization DPI: {config.visualization.dpi}")
        print(f"Output directory: {config.files.output_directory}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 FIXED ADVANCED RECOMMENDER TEST SUITE")
    print("=" * 60)
    
    success = True
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            success = quick_demo()
        elif sys.argv[1] == '--config':
            success = test_custom_configurations()
        elif sys.argv[1] == '--full':
            success = test_visualization_functionality()
        else:
            print("Usage: python test_fixed_visualizations.py [--demo|--config|--full]")
            sys.exit(1)
    else:
        # Default: run all tests
        success &= quick_demo()
        success &= test_custom_configurations() 
        success &= test_visualization_functionality()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("🎯 The fixed advanced recommender is working correctly with visualizations!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1) 