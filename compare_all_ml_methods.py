"""
Compare All ML Methods Individually
This script runs each ML optimization method separately and shows detailed comparisons
"""

from enhanced_financial_recommender_with_ml import MLEnhancedFinancialRecommender
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_all_ml_methods_individually():
    """
    Run each ML method individually and compare their results
    """
    print("🔬 COMPREHENSIVE ML METHOD COMPARISON")
    print("=" * 80)
    
    methods = ['supervised', 'pca', 'genetic', 'multi_objective']
    method_names = ['Supervised Learning', 'PCA Weights', 'Genetic Algorithm', 'Multi-Objective']
    
    results = {}
    
    # Run each method individually
    for method, method_name in zip(methods, method_names):
        print(f"\n🤖 RUNNING {method_name.upper()}")
        print("=" * 60)
        
        try:
            # Create recommender for this specific method
            recommender = MLEnhancedFinancialRecommender(
                'joined_user_table.csv', 
                use_ml_weights=True
            )
            
            # Run analysis with this specific method
            recommendations = recommender.run_ml_enhanced_analysis(
                optimization_method=method
            )
            
            # Store results
            results[method] = {
                'recommender': recommender,
                'recommendations': recommendations,
                'weights': recommender.ml_weights,
                'segments': recommendations['enhanced_segment'].value_counts()
            }
            
            # Save method-specific results
            recommendations.to_csv(f'recommendations_{method}.csv', index=False)
            print(f"✅ Saved: recommendations_{method}.csv")
            
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            results[method] = None
    
    # Create comprehensive comparison
    create_comprehensive_comparison(results, methods, method_names)
    
    return results

def create_comprehensive_comparison(results, methods, method_names):
    """
    Create detailed comparison visualizations and analysis
    """
    print("\n📊 COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 80)
    
    # 1. Weight Comparison Table
    print("\n1️⃣ WEIGHT COMPARISON:")
    print("-" * 50)
    print(f"{'Method':<20} {'Click Rate':<12} {'Avg Time':<12} {'Interactions':<12}")
    print("-" * 60)
    
    # Manual baseline
    print(f"{'Manual (Baseline)':<20} {0.4:<12.3f} {0.3:<12.3f} {0.3:<12.3f}")
    
    # ML methods
    for method, method_name in zip(methods, method_names):
        if results[method] and results[method]['weights']:
            weights = results[method]['weights']
            print(f"{method_name:<20} {weights['click_rate']:<12.3f} {weights['avg_time_viewed']:<12.3f} {weights['total_interactions']:<12.3f}")
        else:
            print(f"{method_name:<20} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12}")
    
    # 2. Segment Distribution Comparison
    print("\n2️⃣ SEGMENT DISTRIBUTION COMPARISON:")
    print("-" * 50)
    
    successful_methods = [(method, method_name) for method, method_name in zip(methods, method_names) if results[method]]
    
    if successful_methods:
        # Get all unique segments across all methods
        all_segments = set()
        for method, _ in successful_methods:
            all_segments.update(results[method]['segments'].index)
        all_segments = sorted(list(all_segments))
        
        # Create comparison table
        print(f"{'Segment':<25}", end="")
        for _, method_name in successful_methods:
            print(f"{method_name[:15]:<15}", end="")
        print()
        print("-" * (25 + 15 * len(successful_methods)))
        
        for segment in all_segments:
            print(f"{segment:<25}", end="")
            for method, _ in successful_methods:
                count = results[method]['segments'].get(segment, 0)
                print(f"{count:<15}", end="")
            print()
    
    # 3. Engagement Score Analysis
    print("\n3️⃣ ENGAGEMENT SCORE IMPACT ANALYSIS:")
    print("-" * 50)
    
    if successful_methods:
        # Compare engagement score distributions
        engagement_stats = {}
        
        for method, method_name in successful_methods:
            recommender = results[method]['recommender']
            engagement_scores = recommender.user_features['engagement_score']
            
            engagement_stats[method_name] = {
                'mean': engagement_scores.mean(),
                'std': engagement_scores.std(),
                'min': engagement_scores.min(),
                'max': engagement_scores.max()
            }
        
        print(f"{'Method':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)
        
        for method_name, stats in engagement_stats.items():
            print(f"{method_name:<20} {stats['mean']:<10.3f} {stats['std']:<10.3f} {stats['min']:<10.3f} {stats['max']:<10.3f}")
    
    # 4. Create visual comparison
    create_method_comparison_visualization(results, methods, method_names)

def create_method_comparison_visualization(results, methods, method_names):
    """
    Create visual comparison of all methods
    """
    successful_results = [(method, method_name, results[method]) for method, method_name in zip(methods, method_names) if results[method]]
    
    if len(successful_results) < 2:
        print("⚠️  Not enough successful methods for visualization comparison")
        return
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ML Method Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. Weight comparison bar chart
    ax1 = axes[0, 0]
    method_labels = [name[:10] for _, name, _ in successful_results]
    click_weights = [result['weights']['click_rate'] for _, _, result in successful_results]
    time_weights = [result['weights']['avg_time_viewed'] for _, _, result in successful_results]
    interaction_weights = [result['weights']['total_interactions'] for _, _, result in successful_results]
    
    x = np.arange(len(method_labels))
    width = 0.25
    
    ax1.bar(x - width, click_weights, width, label='Click Rate', alpha=0.7)
    ax1.bar(x, time_weights, width, label='Avg Time', alpha=0.7)
    ax1.bar(x + width, interaction_weights, width, label='Interactions', alpha=0.7)
    
    ax1.set_xlabel('ML Methods')
    ax1.set_ylabel('Weight Values')
    ax1.set_title('Learned Weight Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Segment distribution comparison
    ax2 = axes[0, 1]
    segment_data = {}
    for method, method_name, result in successful_results:
        segment_data[method_name[:10]] = result['segments']
    
    # Get top 5 segments
    all_segments = set()
    for segments in segment_data.values():
        all_segments.update(segments.index[:5])  # Top 5 segments
    all_segments = list(all_segments)[:5]
    
    bottom = np.zeros(len(successful_results))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_segments)))
    
    for i, segment in enumerate(all_segments):
        values = [segment_data[method_name].get(segment, 0) for _, method_name, _ in successful_results]
        ax2.bar(method_labels, values, bottom=bottom, label=segment[:15], color=colors[i], alpha=0.7)
        bottom += values
    
    ax2.set_xlabel('ML Methods')
    ax2.set_ylabel('Number of Users')
    ax2.set_title('Segment Distribution by Method')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. Engagement score distributions
    ax3 = axes[1, 0]
    engagement_data = []
    labels = []
    
    for method, method_name, result in successful_results:
        engagement_scores = result['recommender'].user_features['engagement_score']
        engagement_data.append(engagement_scores.values)
        labels.append(method_name[:10])
    
    box_plot = ax3.boxplot(engagement_data, labels=labels, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        
    ax3.set_xlabel('ML Methods')
    ax3.set_ylabel('Engagement Score')
    ax3.set_title('Engagement Score Distribution by Method')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Weight deviation from manual baseline
    ax4 = axes[1, 1]
    manual_weights = [0.4, 0.3, 0.3]
    feature_names = ['Click Rate', 'Avg Time', 'Interactions']
    
    for i, (method, method_name, result) in enumerate(successful_results):
        ml_weights = [result['weights']['click_rate'], result['weights']['avg_time_viewed'], result['weights']['total_interactions']]
        deviations = [ml - manual for ml, manual in zip(ml_weights, manual_weights)]
        
        x_pos = np.arange(len(feature_names)) + i * 0.15
        ax4.bar(x_pos, deviations, 0.15, label=method_name[:10], alpha=0.7)
    
    ax4.set_xlabel('Engagement Features')
    ax4.set_ylabel('Weight Deviation from Manual')
    ax4.set_title('Weight Changes vs Manual Baseline')
    ax4.set_xticks(np.arange(len(feature_names)) + 0.15)
    ax4.set_xticklabels(feature_names)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_methods_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: ml_methods_comprehensive_comparison.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("🚀 RUNNING COMPREHENSIVE ML METHOD COMPARISON")
    print("This will run each ML method individually and compare results")
    print()
    
    try:
        results = compare_all_ml_methods_individually()
        
        print("\n🎉 COMPARISON COMPLETE!")
        print("=" * 50)
        print("\n📁 Files Generated:")
        print("  • recommendations_supervised.csv")
        print("  • recommendations_pca.csv") 
        print("  • recommendations_genetic.csv")
        print("  • recommendations_multi_objective.csv")
        print("  • ml_methods_comprehensive_comparison.png")
        
        print("\n💡 INSIGHTS:")
        print("  • Compare weight differences between methods")
        print("  • See how segmentation changes with different approaches")
        print("  • Analyze engagement score distributions")
        print("  • Identify which method works best for your data")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Run enhanced_financial_recommender.py first to generate the data")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all required files are available") 