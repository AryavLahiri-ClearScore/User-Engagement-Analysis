import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from enhanced_financial_recommender import FinanciallyAwareRecommender

def create_dti_distribution_analysis():
    """Create comprehensive DTI distribution visualizations"""
    
    # Load data
    df = pd.read_csv('joined_user_table.csv')
    
    # Get unique users with their DTI ratios
    user_dti = df.groupby('user_id')['dti_ratio'].first()
    
    # Filter for users with DTI > 35%
    high_dti_users = user_dti[user_dti > 0.35]
    
    print("📊 DTI DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Total users: {len(user_dti)}")
    print(f"Users with DTI > 35%: {len(high_dti_users)}")
    print(f"Percentage with DTI > 35%: {len(high_dti_users)/len(user_dti)*100:.1f}%")
    print()
    
    # Statistics comparison
    print("STATISTICS COMPARISON:")
    print("-" * 40)
    print(f"ALL USERS:")
    print(f"  Mean DTI: {user_dti.mean():.1%}")
    print(f"  Median DTI: {user_dti.median():.1%}")
    print(f"  Std Dev: {user_dti.std():.1%}")
    print(f"  Min DTI: {user_dti.min():.1%}")
    print(f"  Max DTI: {user_dti.max():.1%}")
    print()
    print(f"USERS WITH DTI > 35%:")
    print(f"  Mean DTI: {high_dti_users.mean():.1%}")
    print(f"  Median DTI: {high_dti_users.median():.1%}")
    print(f"  Std Dev: {high_dti_users.std():.1%}")
    print(f"  Min DTI: {high_dti_users.min():.1%}")
    print(f"  Max DTI: {high_dti_users.max():.1%}")
    print()
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DTI Distribution Analysis: All Users vs DTI > 35%', fontsize=16, fontweight='bold')
    
    # 1. Side-by-side histograms
    ax1.hist(user_dti * 100, bins=20, alpha=0.7, color='skyblue', 
             label=f'All Users (n={len(user_dti)})', edgecolor='black')
    ax1.hist(high_dti_users * 100, bins=20, alpha=0.7, color='red', 
             label=f'DTI > 35% (n={len(high_dti_users)})', edgecolor='black')
    ax1.axvline(35, color='orange', linestyle='--', linewidth=2, label='35% Threshold')
    ax1.set_xlabel('DTI Ratio (%)')
    ax1.set_ylabel('Number of Users')
    ax1.set_title('DTI Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plots comparison
    box_data = [user_dti * 100, high_dti_users * 100]
    box_labels = ['All Users', 'DTI > 35%']
    box_plot = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('skyblue')
    box_plot['boxes'][1].set_facecolor('red')
    ax2.axhline(35, color='orange', linestyle='--', linewidth=2, label='35% Threshold')
    ax2.set_ylabel('DTI Ratio (%)')
    ax2.set_title('DTI Distribution Box Plots')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Density plots (overlaid)
    ax3.hist(user_dti * 100, bins=30, density=True, alpha=0.6, color='skyblue', 
             label='All Users', edgecolor='black')
    ax3.hist(high_dti_users * 100, bins=15, density=True, alpha=0.6, color='red', 
             label='DTI > 35%', edgecolor='black')
    ax3.axvline(35, color='orange', linestyle='--', linewidth=2, label='35% Threshold')
    ax3.axvline(50, color='darkred', linestyle='--', linewidth=2, label='50% Threshold (High DTI)')
    ax3.set_xlabel('DTI Ratio (%)')
    ax3.set_ylabel('Density')
    ax3.set_title('DTI Density Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_all = np.sort(user_dti * 100)
    sorted_high = np.sort(high_dti_users * 100)
    
    ax4.plot(sorted_all, np.arange(1, len(sorted_all) + 1) / len(sorted_all) * 100, 
             label='All Users', linewidth=2, color='skyblue')
    ax4.plot(sorted_high, np.arange(1, len(sorted_high) + 1) / len(sorted_high) * 100, 
             label='DTI > 35%', linewidth=2, color='red')
    ax4.axvline(35, color='orange', linestyle='--', linewidth=2, label='35% Threshold')
    ax4.axvline(50, color='darkred', linestyle='--', linewidth=2, label='50% Threshold')
    ax4.set_xlabel('DTI Ratio (%)')
    ax4.set_ylabel('Cumulative Percentage')
    ax4.set_title('Cumulative DTI Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dti_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'dti_distribution_analysis.png'")
    plt.show()
    
    # Create engagement vs financial correlation analysis
    print("\n🔗 ENGAGEMENT vs FINANCIAL CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Initialize recommender and create enhanced features
    recommender = FinanciallyAwareRecommender('joined_user_table.csv')
    user_features = recommender.create_enhanced_user_features()
    
    # Select engagement and financial metrics for correlation
    engagement_metrics = ['engagement_score', 'click_rate', 'avg_time_viewed', 
                         'total_interactions', 'unique_content_viewed']
    financial_metrics = ['financial_health_score', 'credit_score', 'dti_ratio', 
                        'income', 'total_debt', 'missed_payments']
    
    # Combine all metrics for correlation analysis
    correlation_metrics = engagement_metrics + financial_metrics
    correlation_data = user_features[correlation_metrics]
    
    # Calculate correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Create correlation heatmap
    fig2, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create heatmap with custom formatting
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)  # Mask upper triangle
    heatmap = sns.heatmap(correlation_matrix, 
                         mask=mask,
                         annot=True, 
                         cmap='RdBu_r', 
                         center=0,
                         square=True,
                         fmt='.2f',
                         cbar_kws={"shrink": .8},
                         ax=ax)
    
    ax.set_title('Engagement vs Financial Metrics Correlation Matrix', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('engagement_financial_correlation.png', dpi=300, bbox_inches='tight')
    print("✅ Correlation matrix saved as 'engagement_financial_correlation.png'")
    plt.show()
    
    # Print key correlations
    print("\nKEY CORRELATIONS (|r| > 0.3):")
    print("-" * 50)
    
    # Find significant correlations between engagement and financial metrics
    for eng_metric in engagement_metrics:
        for fin_metric in financial_metrics:
            corr_value = correlation_matrix.loc[eng_metric, fin_metric]
            if abs(corr_value) > 0.3:
                direction = "positive" if corr_value > 0 else "negative"
                strength = "strong" if abs(corr_value) > 0.7 else "moderate"
                print(f"• {eng_metric} ↔ {fin_metric}: {corr_value:.3f} ({strength} {direction})")
    
    # Highlight surprising or concerning correlations
    print("\n⚠️  NOTABLE FINDINGS:")
    print("-" * 30)
    
    # Check for counterintuitive correlations
    engagement_financial_corr = correlation_matrix.loc[engagement_metrics, financial_metrics]
    
    # DTI vs engagement correlations
    dti_eng_corr = engagement_financial_corr.loc[:, 'dti_ratio']
    positive_dti_corr = dti_eng_corr[dti_eng_corr > 0.1]
    if len(positive_dti_corr) > 0:
        print("🚨 Positive DTI correlations (concerning):")
        for metric, corr in positive_dti_corr.items():
            print(f"   - Higher {metric} correlates with higher DTI ({corr:.3f})")
    
    # Credit score vs engagement correlations  
    credit_eng_corr = engagement_financial_corr.loc[:, 'credit_score']
    negative_credit_corr = credit_eng_corr[credit_eng_corr < -0.1]
    if len(negative_credit_corr) > 0:
        print("🚨 Negative credit score correlations (concerning):")
        for metric, corr in negative_credit_corr.items():
            print(f"   - Higher {metric} correlates with lower credit score ({corr:.3f})")
    
    # Strong positive correlations with financial health
    fin_health_corr = engagement_financial_corr.loc[:, 'financial_health_score']
    strong_positive = fin_health_corr[fin_health_corr > 0.3]
    if len(strong_positive) > 0:
        print("✅ Strong positive financial health correlations:")
        for metric, corr in strong_positive.items():
            print(f"   - Higher {metric} correlates with better financial health ({corr:.3f})")
    
    print()
    
    # Category breakdown
    print("\nDTI CATEGORY BREAKDOWN:")
    print("-" * 40)
    categories = {
        'Healthy (< 25%)': (user_dti < 0.25).sum(),
        'Moderate (25-34%)': ((user_dti >= 0.25) & (user_dti < 0.35)).sum(),
        'Elevated (35-49%)': ((user_dti >= 0.35) & (user_dti < 0.50)).sum(),
        'High DTI (≥ 50%)': (user_dti >= 0.50).sum()
    }
    
    for category, count in categories.items():
        percentage = count / len(user_dti) * 100
        print(f"{category}: {count} users ({percentage:.1f}%)")
    
    print()
    print("KEY INSIGHTS:")
    print("-" * 40)
    
    # Calculate percentiles
    p25 = np.percentile(user_dti * 100, 25)
    p50 = np.percentile(user_dti * 100, 50)
    p75 = np.percentile(user_dti * 100, 75)
    p90 = np.percentile(user_dti * 100, 90)
    
    print(f"• 25th percentile DTI: {p25:.1f}%")
    print(f"• 50th percentile DTI: {p50:.1f}%")
    print(f"• 75th percentile DTI: {p75:.1f}%")
    print(f"• 90th percentile DTI: {p90:.1f}%")
    print()
    
    if len(high_dti_users) > len(user_dti) * 0.5:
        print(f"⚠️  WARNING: {len(high_dti_users)/len(user_dti)*100:.1f}% of users have DTI > 35%")
        print("   This suggests the 35% threshold may be too low for this population.")
    else:
        print(f"✅ {len(high_dti_users)/len(user_dti)*100:.1f}% of users have DTI > 35%")
        print("   This seems like a reasonable threshold for identifying high-risk users.")
    
    return user_dti, high_dti_users, correlation_matrix, user_features

if __name__ == "__main__":
    all_users, high_dti_users, corr_matrix, features = create_dti_distribution_analysis()
    
    print("\n📈 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("• dti_distribution_analysis.png - DTI distribution comparison")
    print("• engagement_financial_correlation.png - Correlation heatmap")
    print()
    print("Key insights:")
    print(f"• {len(high_dti_users)} out of {len(all_users)} users have DTI > 35%")
    print(f"• Correlation analysis reveals relationships between engagement and financial health")
    print(f"• Use correlation matrix to understand user behavior patterns") 