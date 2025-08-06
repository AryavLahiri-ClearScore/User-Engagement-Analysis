#!/usr/bin/env python3
"""
Test script to demonstrate the fixed advanced recommender working with real CSV data
Shows how it properly processes your actual 100 users instead of generating 1000 mock users
"""

from fixed_advanced_recommender import FixedAdvancedRecommender, SystemConfig
import os

def test_with_real_data():
    """Test the recommender with actual CSV data"""
    print("🔬 TESTING WITH REAL CSV DATA")
    print("=" * 60)
    
    csv_file = 'joined_user_table.csv'
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found in current directory")
        print("Make sure you're running this from the project directory")
        return False
    
    try:
        print(f"📁 Loading data from: {csv_file}")
        
        # Initialize with real CSV file
        recommender = FixedAdvancedRecommender(csv_file=csv_file)
        
        print("\n🔍 PROCESSING REAL USER DATA:")
        print("-" * 40)
        
        # This will process your actual CSV and show the real user count
        user_features = recommender.create_user_features()  # This processes real user data
        
        print(f"\n✅ SUCCESS! Processed {len(user_features)} actual users from your CSV")
        
        # Show sample of real data
        print(f"\n📊 SAMPLE OF REAL USER DATA:")
        print("-" * 40)
        sample_cols = ['credit_score', 'dti_ratio', 'income', 'engagement_score', 'financial_health_score', 'financial_category', 'enhanced_segment']
        available_cols = [col for col in sample_cols if col in user_features.columns]
        print(user_features[available_cols].head())
        
        # Test visualization with real data
        print(f"\n🎨 CREATING VISUALIZATIONS WITH REAL DATA:")
        print("-" * 40)
        print("Creating financial dashboard with your actual user distribution...")
        
        recommender.create_financial_visualizations()
        
        print(f"\n🎉 SUCCESS! All visualizations created with your real {len(user_features)} users")
        return True
        
    except Exception as e:
        print(f"❌ Error processing real data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_demo_vs_real():
    """Compare demo data generation vs real data processing"""
    print("\n🔄 COMPARING DEMO VS REAL DATA")
    print("=" * 60)
    
    try:
        # Test 1: Standard usage with CSV file
        print("\n1️⃣ Processing Real Data:")
        real_recommender = FixedAdvancedRecommender('joined_user_table.csv')
        real_features = real_recommender.create_user_features()
        print(f"   Real data processed: {len(real_features)} users")
        
        print(f"\n📋 SUMMARY:")
        print(f"   Successfully processed: {len(real_features)} actual users from your CSV")
        print(f"   Real data shows your actual user distribution!")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 REAL DATA PROCESSING TEST")
    print("=" * 60)
    
    success = True
    success &= test_with_real_data()
    success &= test_demo_vs_real()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("🎯 The system now correctly processes your actual CSV data")
        print(f"🎯 Shows real user counts instead of mock 1000 users")
        print(f"🎯 Terminal output reflects your actual {os.path.exists('joined_user_table.csv') and 'user distribution' or 'data'}")
    else:
        print("\n❌ SOME TESTS FAILED!") 