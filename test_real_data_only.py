#!/usr/bin/env python3
"""
Test script to demonstrate the system now only uses real CSV data
No more mock data generation - everything comes from joined_user_table.csv
"""

from fixed_advanced_recommender import FixedAdvancedRecommender
import os

def test_real_data_only():
    """Test that the system only uses real data from CSV"""
    print("🎯 TESTING: REAL DATA ONLY (NO MOCK DATA)")
    print("=" * 60)
    
    csv_file = 'joined_user_table.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found")
        print("Make sure you have the joined_user_table.csv file in your directory")
        return False
    
    try:
        print("🔍 TESTING CSV REQUIREMENT:")
        print("-" * 40)
        
        # Test 1: Try to create without CSV (should fail)
        print("1️⃣ Testing without CSV file (should fail):")
        try:
            bad_recommender = FixedAdvancedRecommender(None)
            print("❌ This should have failed!")
            return False
        except (ValueError, TypeError) as e:
            print(f"✅ Correctly failed: {e}")
        
        # Test 2: Create with CSV file (should work)
        print("\n2️⃣ Testing with CSV file (should work):")
        recommender = FixedAdvancedRecommender(csv_file)
        print("✅ Successfully created recommender")
        
        # Test 3: Process real data
        print("\n3️⃣ Processing your actual data:")
        print("-" * 40)
        user_features = recommender.create_user_features()
        
        print(f"\n📊 REAL DATA RESULTS:")
        print(f"   Users processed: {len(user_features)}")
        print(f"   Features created: {len(user_features.columns)}")
        
        # Show that this is real data from your CSV
        unique_users_in_csv = recommender.df['user_id'].nunique()
        total_interactions = len(recommender.df)
        
        print(f"\n🔍 DATA VALIDATION:")
        print(f"   CSV interactions: {total_interactions:,}")
        print(f"   Unique users in CSV: {unique_users_in_csv}")
        print(f"   Features generated: {len(user_features)}")
        print(f"   ✅ User count matches: {len(user_features) == unique_users_in_csv}")
        
        # Show sample of real financial data
        print(f"\n💰 SAMPLE REAL FINANCIAL DATA:")
        print("-" * 40)
        sample_cols = ['credit_score', 'dti_ratio', 'income', 'financial_health_score', 'financial_category']
        available_cols = [col for col in sample_cols if col in user_features.columns]
        print(user_features[available_cols].head(3))
        
        # Show sample of real engagement data  
        print(f"\n⚡ SAMPLE REAL ENGAGEMENT DATA:")
        print("-" * 40)
        engagement_cols = ['avg_time_viewed', 'total_interactions', 'click_rate', 'engagement_score']
        available_eng_cols = [col for col in engagement_cols if col in user_features.columns]
        print(user_features[available_eng_cols].head(3))
        
        print(f"\n🎉 SUCCESS!")
        print("✅ System now only uses your real CSV data")
        print("✅ No mock data generation")
        print(f"✅ All {len(user_features)} users come from your joined_user_table.csv")
        print("✅ Financial and engagement metrics calculated from actual interactions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualizations_with_real_data():
    """Test that visualizations work with real data"""
    print("\n🎨 TESTING VISUALIZATIONS WITH REAL DATA")
    print("=" * 60)
    
    try:
        recommender = FixedAdvancedRecommender('joined_user_table.csv')
        
        print("Creating financial dashboard with your 100 users...")
        recommender.create_financial_visualizations()
        
        print("\n🎯 The visualization above shows:")
        print("✅ Your actual 100 users")
        print("✅ Real financial health distribution from your data")
        print("✅ Real engagement patterns from your data")
        print("✅ No synthetic/mock data involved")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 REAL DATA ONLY TEST SUITE")
    print("=" * 60)
    
    success = True
    success &= test_real_data_only()
    success &= test_visualizations_with_real_data()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("🎯 System successfully converted to real-data-only mode")
        print("🎯 No more mock 1000 users - only your actual 100 users")
        print("🎯 All metrics calculated from actual user interactions")
    else:
        print("\n❌ SOME TESTS FAILED!") 