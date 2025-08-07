"""
Test script to verify that the analysis produces identical results on multiple runs
"""

import pandas as pd
import numpy as np
from compare_manual_vs_ml_analysis import ManualVsMLComparison

def test_reproducibility():
    """Test that running the analysis multiple times gives identical results"""
    print("🔬 TESTING REPRODUCIBILITY")
    print("=" * 50)
    
    results = []
    
    # Run the analysis 3 times
    for run in range(3):
        print(f"\n📊 RUN {run + 1}")
        print("-" * 20)
        
        # Create fresh analyzer instance
        analyzer = ManualVsMLComparison('joined_user_table.csv')
        
        # Run only the core analysis (skip visualizations for speed)
        training_data = analyzer.create_real_training_data()
        model_performance = analyzer.train_ml_recommendation_models(training_data)
        manual_results = analyzer.calculate_manual_weights_analysis()
        
        # Store key results for comparison
        results.append({
            'run': run + 1,
            'training_samples': len(training_data),
            'recommendation_accuracy': model_performance['recommendation_accuracy'],
            'priority_accuracy': model_performance['priority_accuracy'],
            'urgency_accuracy': model_performance['urgency_accuracy'],
            'manual_users': len(manual_results),
            'avg_manual_financial_score': manual_results['manual_financial_health_score'].mean(),
            'avg_manual_engagement_score': manual_results['manual_engagement_score'].mean(),
            'manual_excellent_count': (manual_results['manual_financial_category'] == 'Excellent').sum(),
            'manual_premium_engaged_count': (manual_results['manual_enhanced_segment'] == 'Premium_Engaged').sum()
        })
        
        print(f"   ✅ Run {run + 1} completed")
    
    # Compare results
    print(f"\n🔍 REPRODUCIBILITY CHECK")
    print("=" * 50)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Check if all runs are identical
    all_identical = True
    numeric_columns = ['recommendation_accuracy', 'priority_accuracy', 'urgency_accuracy', 
                      'avg_manual_financial_score', 'avg_manual_engagement_score',
                      'manual_excellent_count', 'manual_premium_engaged_count']
    
    for col in numeric_columns:
        values = df_results[col]
        if not all(abs(values - values.iloc[0]) < 1e-10):  # Allow tiny floating point differences
            all_identical = False
            print(f"❌ VARIATION in {col}: {values.tolist()}")
    
    if all_identical:
        print(f"\n✅ SUCCESS: All runs produced IDENTICAL results!")
        print(f"🎯 The analysis is now fully reproducible!")
    else:
        print(f"\n❌ ISSUE: Results vary between runs!")
        print(f"🔧 Check for unseeded random operations!")
    
    return all_identical

if __name__ == "__main__":
    test_reproducibility() 