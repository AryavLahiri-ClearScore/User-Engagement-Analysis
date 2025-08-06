#!/usr/bin/env python3
"""
Debug script to identify why visualizations are blank
This will run all the diagnostic functions to find data issues
"""

from fixed_advanced_recommender import FixedAdvancedRecommender

def debug_blank_visualizations():
    """Run comprehensive debugging for blank visualization issues"""
    print("🔍 DEBUGGING BLANK VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Initialize with your CSV
        print("1️⃣ Initializing recommender...")
        recommender = FixedAdvancedRecommender('joined_user_table.csv')
        
        # Create user features (this will show content types and data validation)
        print("\n2️⃣ Creating user features...")
        user_features = recommender.create_user_features()
        
        # Run the built-in diagnosis
        print("\n3️⃣ Running visualization diagnosis...")
        recommender.diagnose_blank_visualizations()
        
        # Try creating one visualization to see what happens
        print("\n4️⃣ Testing financial visualization...")
        try:
            recommender.create_financial_visualizations()
            print("✅ Financial visualization completed")
        except Exception as e:
            print(f"❌ Financial visualization failed: {e}")
        
        # Test clustering visualization
        print("\n5️⃣ Testing clustering visualization...")
        try:
            recommender.visualize_financial_clustering()
            print("✅ Clustering visualization completed")
        except Exception as e:
            print(f"❌ Clustering visualization failed: {e}")
        
        # Test correlation visualization
        print("\n6️⃣ Testing correlation visualization...")
        try:
            recommender.create_engagement_financial_correlation()
            print("✅ Correlation visualization completed")
        except Exception as e:
            print(f"❌ Correlation visualization failed: {e}")
        
        print("\n🎯 DEBUGGING COMPLETE!")
        print("Check the output above for:")
        print("• Missing columns or features")
        print("• Data with all zeros or very small ranges")
        print("• Content type mismatches")
        print("• Scaling or normalization issues")
        
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_blank_visualizations() 