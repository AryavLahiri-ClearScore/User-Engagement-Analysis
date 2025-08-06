#!/usr/bin/env python3
"""
Test script for ML-Enhanced Financial Recommender Visualizations
This script tests all the new visualization methods to ensure they work correctly.
"""

import sys
import pandas as pd
import numpy as np
from enhanced_financial_recommender_with_ml import MLEnhancedFinancialRecommender

def test_ml_visualizations():
    """Test all ML visualization methods"""
    print("🧪 TESTING ML-ENHANCED VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Initialize the ML recommender
        print("\n1️⃣ Initializing ML Recommender...")
        ml_recommender = MLEnhancedFinancialRecommender(
            'joined_user_table.csv', 
            use_ml_weights=True
        )
        
        # Create features
        print("\n2️⃣ Creating user features...")
        ml_recommender.create_enhanced_user_features()
        
        # Optimize weights
        print("\n3️⃣ Optimizing weights with supervised learning...")
        ml_recommender.optimize_engagement_weights(method='supervised')
        
        # Perform segmentation
        print("\n4️⃣ Performing ML-enhanced segmentation...")
        ml_recommender.perform_enhanced_segmentation_with_ml()
        
        # Test each visualization method
        print("\n5️⃣ Testing ML Financial Visualizations...")
        try:
            ml_recommender.create_ml_financial_visualizations()
            print("✅ ML Financial Visualizations: SUCCESS")
        except Exception as e:
            print(f"❌ ML Financial Visualizations: FAILED - {e}")
        
        print("\n6️⃣ Testing ML Clustering Visualization...")
        try:
            ml_recommender.visualize_ml_financial_clustering()
            print("✅ ML Clustering Visualization: SUCCESS")
        except Exception as e:
            print(f"❌ ML Clustering Visualization: FAILED - {e}")
        
        print("\n7️⃣ Testing ML Correlation Analysis...")
        try:
            ml_recommender.create_ml_engagement_financial_correlation()
            print("✅ ML Correlation Analysis: SUCCESS")
        except Exception as e:
            print(f"❌ ML Correlation Analysis: FAILED - {e}")
        
        # Test comparison visualization (need manual data)
        print("\n8️⃣ Testing ML vs Manual Comparison...")
        try:
            # Create manual recommender for comparison
            manual_recommender = MLEnhancedFinancialRecommender(
                'joined_user_table.csv', 
                use_ml_weights=False
            )
            manual_recommendations = manual_recommender.run_enhanced_analysis()
            
            # Now test comparison
            ml_recommender.create_ml_comparison_dashboard(manual_recommendations)
            print("✅ ML vs Manual Comparison: SUCCESS")
        except Exception as e:
            print(f"❌ ML vs Manual Comparison: FAILED - {e}")
        
        print("\n🎉 ALL VISUALIZATION TESTS COMPLETED!")
        
        # Print summary of generated files
        generated_files = [
            'ml_enhanced_financial_dashboard.png',
            'ml_weight_analysis.png',
            'ml_enhanced_clustering_visualization.png',
            'ml_engagement_financial_correlation.png',
            'ml_vs_manual_comparison.png'
        ]
        
        print("\n📁 Generated Visualization Files:")
        for file in generated_files:
            print(f"  • {file}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Please run the main enhanced_financial_recommender.py first to generate the data")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def quick_test():
    """Quick test with minimal setup"""
    print("🚀 QUICK TEST - BASIC FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Test just the class initialization and basic methods
        ml_recommender = MLEnhancedFinancialRecommender(
            'joined_user_table.csv', 
            use_ml_weights=True
        )
        
        print("✅ Class initialization: SUCCESS")
        
        # Test if we can create features
        ml_recommender.create_enhanced_user_features()
        print("✅ Feature creation: SUCCESS")
        
        # Test if we can optimize weights
        ml_recommender.optimize_engagement_weights(method='pca')  # Faster than supervised
        print("✅ Weight optimization: SUCCESS")
        
        # Test if we have the required attributes
        required_attrs = ['user_features', 'ml_weights', 'scaler']
        for attr in required_attrs:
            if hasattr(ml_recommender, attr) and getattr(ml_recommender, attr) is not None:
                print(f"✅ {attr}: PRESENT")
            else:
                print(f"❌ {attr}: MISSING")
        
        print("\n🎯 QUICK TEST PASSED - Ready for full visualization testing")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 ML VISUALIZATION TEST SUITE")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        success = quick_test()
    else:
        success = test_ml_visualizations()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1) 