"""
Comprehensive Manual vs ML Comparison Analysis
Creates visualizations and CSV files comparing manual weights vs ML-learned weights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from fully_ml_financial_recommender import FullyMLFinancialRecommender
import warnings
warnings.filterwarnings('ignore')

class ManualVsMLComparison:
    """Compare manual weight system vs ML-learned weights"""
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.manual_results = None
        self.ml_results = None
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def calculate_manual_weights_analysis(self):
        """Calculate user features using manual weights"""
        print("🔧 CALCULATING MANUAL WEIGHTS ANALYSIS")
        print("=" * 60)
        
        # Create user-level aggregated data
        user_financial = self.df.groupby('user_id').agg({
            'credit_score': 'first',
            'dti_ratio': 'first', 
            'income': 'first',
            'total_debt': 'first',
            'missed_payments': 'first',
            'has_ccj': 'first',
            'has_mortgage': 'first',
            'has_car': 'first'
        }).reset_index()
        
        # Aggregate engagement data per user
        user_engagement = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        user_engagement.columns = ['user_id', 'avg_time_viewed', 'total_time_viewed', 'total_interactions',
                                  'click_rate', 'total_clicks', 'unique_content_viewed']
        
        # Merge data
        user_df = pd.merge(user_financial, user_engagement, on='user_id')
        
        # Calculate manual financial health scores
        manual_financial_weights = {
            'credit_component': 0.30,
            'dti_component': 0.25, 
            'missed_payments_component': 0.15,
            'income_component': 0.15,
            'ccj_component': 0.10,
            'asset_component': 0.05
        }
        
        # Calculate components
        credit_comp = np.minimum(user_df['credit_score'] / 1000, 1.0)
        dti_comp = np.maximum(0, 1 - user_df['dti_ratio'])
        missed_comp = np.maximum(0, 1 - (user_df['missed_payments'] / 10))
        income_comp = np.minimum(user_df['income'] / 100000, 1.0)
        ccj_comp = 1 - user_df['has_ccj']
        asset_comp = user_df['has_mortgage'] * 0.6 + user_df['has_car'] * 0.4
        
        # Calculate manual financial health scores
        user_df['manual_financial_health_score'] = (
            credit_comp * manual_financial_weights['credit_component'] +
            dti_comp * manual_financial_weights['dti_component'] +
            missed_comp * manual_financial_weights['missed_payments_component'] +
            income_comp * manual_financial_weights['income_component'] +
            ccj_comp * manual_financial_weights['ccj_component'] +
            asset_comp * manual_financial_weights['asset_component']
        )
        
        # Calculate manual engagement scores
        manual_engagement_weights = {
            'click_rate': 0.4,
            'avg_time_viewed': 0.3,
            'total_interactions': 0.3
        }
        
        # Normalize engagement features
        normalized_time = np.minimum(user_df['avg_time_viewed'] / 60, 1.0)
        normalized_interactions = np.minimum(user_df['total_interactions'] / 12, 1.0)
        
        user_df['manual_engagement_score'] = (
            user_df['click_rate'] * manual_engagement_weights['click_rate'] +
            normalized_time * manual_engagement_weights['avg_time_viewed'] +
            normalized_interactions * manual_engagement_weights['total_interactions']
        )
        
        # Categorize financial health
        def categorize_financial_health(score):
            if score >= 0.8:
                return "Excellent"
            elif score >= 0.65:
                return "Good"
            elif score >= 0.45:
                return "Fair"
            else:
                return "Poor"
        
        user_df['manual_financial_category'] = user_df['manual_financial_health_score'].apply(categorize_financial_health)
        
        # Assign enhanced segments
        def assign_enhanced_segments(row):
            engagement = row['manual_engagement_score']
            financial_cat = row['manual_financial_category']
            dti_ratio = row['dti_ratio']
            missed_payments = row['missed_payments']
            
            if missed_payments >= 2:
                return "Payment_Recovery_Priority"
            elif dti_ratio >= 0.5:
                return "Debt_Management_Priority"
            elif engagement > 0.5:
                if financial_cat == "Excellent":
                    return "Premium_Engaged"
                elif financial_cat in ["Good", "Fair"]:
                    return "Growth_Focused"
                else:
                    return "Recovery_Engaged"
            elif engagement > 0.25:
                if financial_cat == "Excellent":
                    return "Premium_Moderate"
                elif financial_cat in ["Good", "Fair"]:
                    return "Mainstream"
                else:
                    return "Recovery_Moderate"
            else:
                if financial_cat == "Poor":
                    return "Financial_Priority"
                else:
                    return "Activation_Needed"
        
        user_df['manual_enhanced_segment'] = user_df.apply(assign_enhanced_segments, axis=1)
        
        self.manual_results = user_df
        
        print(f"✅ Calculated manual analysis for {len(user_df)} users")
        print(f"📊 Manual Financial Categories: {dict(user_df['manual_financial_category'].value_counts())}")
        print(f"🎯 Manual Enhanced Segments: {dict(user_df['manual_enhanced_segment'].value_counts())}")
        
        return user_df
    
    def run_ml_analysis(self):
        """Run ML analysis using FullyMLFinancialRecommender"""
        print("\n🤖 RUNNING ML ANALYSIS")
        print("=" * 60)
        
        # Create ML recommender
        ml_recommender = FullyMLFinancialRecommender(self.csv_file)
        
        # Run ML analysis
        ml_recommender.run_fully_ml_analysis()
        
        # Get the results
        self.ml_results = ml_recommender.df.copy()
        
        print(f"✅ Completed ML analysis for {len(self.ml_results)} users")
        print(f"📊 ML Financial Categories: {dict(self.ml_results['ml_financial_category'].value_counts())}")
        print(f"🎯 ML Enhanced Segments: {dict(self.ml_results['ml_enhanced_segment'].value_counts())}")
        
        return self.ml_results
    
    def create_comparison_visualizations(self):
        """Create comprehensive comparison visualizations"""
        print("\n🎨 CREATING COMPARISON VISUALIZATIONS")
        print("=" * 60)
        
        # Merge manual and ML results for comparison
        comparison_df = pd.merge(
            self.manual_results[['user_id', 'manual_financial_category', 'manual_enhanced_segment', 
                               'manual_financial_health_score', 'manual_engagement_score']],
            self.ml_results[['user_id', 'ml_financial_category', 'ml_enhanced_segment',
                           'ml_financial_health_score', 'ml_engagement_score']],
            on='user_id'
        )
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Financial Categories Comparison
        plt.subplot(3, 3, 1)
        manual_fin_counts = self.manual_results['manual_financial_category'].value_counts()
        plt.pie(manual_fin_counts.values, labels=manual_fin_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Manual Financial Categories\n(Manual Weights)', fontsize=12, fontweight='bold')
        
        plt.subplot(3, 3, 2)
        ml_fin_counts = self.ml_results['ml_financial_category'].value_counts()
        plt.pie(ml_fin_counts.values, labels=ml_fin_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('ML Financial Categories\n(ML-Learned Weights)', fontsize=12, fontweight='bold')
        
        # 2. Enhanced Segments Comparison
        plt.subplot(3, 3, 4)
        manual_seg_counts = self.manual_results['manual_enhanced_segment'].value_counts()
        plt.barh(range(len(manual_seg_counts)), manual_seg_counts.values, color='skyblue')
        plt.yticks(range(len(manual_seg_counts)), [seg[:15] + '...' if len(seg) > 15 else seg for seg in manual_seg_counts.index])
        plt.title('Manual Enhanced Segments', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Users')
        
        plt.subplot(3, 3, 5)
        ml_seg_counts = self.ml_results['ml_enhanced_segment'].value_counts()
        plt.barh(range(len(ml_seg_counts)), ml_seg_counts.values, color='lightcoral')
        plt.yticks(range(len(ml_seg_counts)), [seg[:15] + '...' if len(seg) > 15 else seg for seg in ml_seg_counts.index])
        plt.title('ML Enhanced Segments', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Users')
        
        # 3. Score Distributions
        plt.subplot(3, 3, 7)
        plt.hist(comparison_df['manual_financial_health_score'], alpha=0.7, label='Manual', bins=20, color='skyblue')
        plt.hist(comparison_df['ml_financial_health_score'], alpha=0.7, label='ML', bins=20, color='lightcoral')
        plt.title('Financial Health Score Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Financial Health Score')
        plt.ylabel('Number of Users')
        plt.legend()
        
        plt.subplot(3, 3, 8)
        plt.hist(comparison_df['manual_engagement_score'], alpha=0.7, label='Manual', bins=20, color='skyblue')
        plt.hist(comparison_df['ml_engagement_score'], alpha=0.7, label='ML', bins=20, color='lightcoral')
        plt.title('Engagement Score Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Engagement Score')
        plt.ylabel('Number of Users')
        plt.legend()
        
        # 4. Scatter Plot Comparison
        plt.subplot(3, 3, 9)
        plt.scatter(comparison_df['manual_financial_health_score'], comparison_df['ml_financial_health_score'], 
                   alpha=0.6, color='purple', s=30)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)  # Diagonal line
        plt.xlabel('Manual Financial Health Score')
        plt.ylabel('ML Financial Health Score')
        plt.title('Manual vs ML Financial Scores\n(Perfect agreement = diagonal line)', fontsize=12, fontweight='bold')
        
        # 5. Category Transition Matrix
        plt.subplot(3, 3, 3)
        # Create transition matrix
        transition_data = []
        categories = ['Poor', 'Fair', 'Good', 'Excellent']
        for manual_cat in categories:
            row = []
            for ml_cat in categories:
                count = len(comparison_df[(comparison_df['manual_financial_category'] == manual_cat) & 
                                        (comparison_df['ml_financial_category'] == ml_cat)])
                row.append(count)
            transition_data.append(row)
        
        sns.heatmap(transition_data, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories, yticklabels=categories)
        plt.title('Category Transition Matrix\n(Manual → ML)', fontsize=12, fontweight='bold')
        plt.xlabel('ML Categories')
        plt.ylabel('Manual Categories')
        
        # 6. Summary Statistics
        plt.subplot(3, 3, 6)
        plt.axis('off')
        
        # Calculate summary stats
        manual_mean_fin = comparison_df['manual_financial_health_score'].mean()
        ml_mean_fin = comparison_df['ml_financial_health_score'].mean()
        manual_mean_eng = comparison_df['manual_engagement_score'].mean()
        ml_mean_eng = comparison_df['ml_engagement_score'].mean()
        
        correlation_fin = comparison_df['manual_financial_health_score'].corr(comparison_df['ml_financial_health_score'])
        correlation_eng = comparison_df['manual_engagement_score'].corr(comparison_df['ml_engagement_score'])
        
        # Count differences
        fin_cat_same = (comparison_df['manual_financial_category'] == comparison_df['ml_financial_category']).sum()
        seg_same = (comparison_df['manual_enhanced_segment'] == comparison_df['ml_enhanced_segment']).sum()
        
        total_users = len(comparison_df)
        
        summary_text = f"""
        COMPARISON SUMMARY
        ═══════════════════
        
        📊 AVERAGE SCORES:
        Manual Financial: {manual_mean_fin:.3f}
        ML Financial:     {ml_mean_fin:.3f}
        
        Manual Engagement: {manual_mean_eng:.3f}
        ML Engagement:     {ml_mean_eng:.3f}
        
        🔗 CORRELATIONS:
        Financial Scores: {correlation_fin:.3f}
        Engagement Scores: {correlation_eng:.3f}
        
        🎯 AGREEMENT RATES:
        Same Financial Category: {fin_cat_same}/{total_users} ({fin_cat_same/total_users*100:.1f}%)
        Same Enhanced Segment:   {seg_same}/{total_users} ({seg_same/total_users*100:.1f}%)
        
        📈 KEY INSIGHTS:
        • Higher correlation = methods agree more
        • Lower correlation = ML found different patterns
        • Category changes show ML discoveries
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('manual_vs_ml_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Created comprehensive comparison visualization: manual_vs_ml_comprehensive_comparison.png")
        
        return comparison_df
    
    def create_detailed_segment_analysis(self, comparison_df):
        """Create detailed segment analysis visualization"""
        print("\n📊 CREATING DETAILED SEGMENT ANALYSIS")
        print("=" * 60)
        
        # Create segment comparison matrix
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Manual segments detailed breakdown
        ax1 = axes[0, 0]
        manual_segments = self.manual_results['manual_enhanced_segment'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(manual_segments)))
        wedges, texts, autotexts = ax1.pie(manual_segments.values, labels=None, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Manual Enhanced Segments Distribution', fontsize=14, fontweight='bold')
        
        # Add legend for manual segments
        ax1.legend(wedges, [f"{seg[:20]}..." if len(seg) > 20 else seg for seg in manual_segments.index],
                  title="Segments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # 2. ML segments detailed breakdown
        ax2 = axes[0, 1]
        ml_segments = self.ml_results['ml_enhanced_segment'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(ml_segments)))
        wedges, texts, autotexts = ax2.pie(ml_segments.values, labels=None, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('ML Enhanced Segments Distribution', fontsize=14, fontweight='bold')
        
        # Add legend for ML segments
        ax2.legend(wedges, [f"{seg[:20]}..." if len(seg) > 20 else seg for seg in ml_segments.index],
                  title="Segments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # 3. Side-by-side comparison
        ax3 = axes[1, 0]
        
        # Get all unique segments
        all_segments = set(list(manual_segments.index) + list(ml_segments.index))
        
        # Prepare data for comparison
        manual_counts = [manual_segments.get(seg, 0) for seg in all_segments]
        ml_counts = [ml_segments.get(seg, 0) for seg in all_segments]
        
        x = np.arange(len(all_segments))
        width = 0.35
        
        ax3.bar(x - width/2, manual_counts, width, label='Manual', color='skyblue', alpha=0.8)
        ax3.bar(x + width/2, ml_counts, width, label='ML', color='lightcoral', alpha=0.8)
        
        ax3.set_xlabel('Enhanced Segments')
        ax3.set_ylabel('Number of Users')
        ax3.set_title('Manual vs ML Segment Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([seg[:10] + '...' if len(seg) > 10 else seg for seg in all_segments], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Segment migration analysis
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate segment migrations
        segment_changes = []
        for manual_seg in manual_segments.index:
            for ml_seg in ml_segments.index:
                count = len(comparison_df[(comparison_df['manual_enhanced_segment'] == manual_seg) & 
                                        (comparison_df['ml_enhanced_segment'] == ml_seg)])
                if count > 0:
                    segment_changes.append({
                        'from': manual_seg,
                        'to': ml_seg,
                        'count': count,
                        'same': manual_seg == ml_seg
                    })
        
        # Sort by count and show top migrations
        segment_changes.sort(key=lambda x: x['count'], reverse=True)
        
        migration_text = "TOP SEGMENT MIGRATIONS\n" + "═" * 30 + "\n\n"
        migration_text += f"{'FROM (Manual)':<20} → {'TO (ML)':<20} | {'COUNT':>5}\n"
        migration_text += "─" * 55 + "\n"
        
        for change in segment_changes[:10]:  # Top 10 migrations
            from_seg = change['from'][:18] + ".." if len(change['from']) > 18 else change['from']
            to_seg = change['to'][:18] + ".." if len(change['to']) > 18 else change['to']
            status = "✓" if change['same'] else "↔"
            migration_text += f"{from_seg:<20} → {to_seg:<20} | {change['count']:>5} {status}\n"
        
        # Add summary stats
        total_same = sum(1 for change in segment_changes if change['same'])
        total_different = len(segment_changes) - total_same
        
        migration_text += f"\n📊 MIGRATION SUMMARY:\n"
        migration_text += f"Users staying in same segment: {comparison_df['manual_enhanced_segment'].eq(comparison_df['ml_enhanced_segment']).sum()}\n"
        migration_text += f"Users changing segments: {len(comparison_df) - comparison_df['manual_enhanced_segment'].eq(comparison_df['ml_enhanced_segment']).sum()}\n"
        migration_text += f"\n✓ = Same segment\n↔ = Different segment"
        
        ax4.text(0.05, 0.95, migration_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('detailed_segment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Created detailed segment analysis: detailed_segment_analysis.png")
    
    def generate_csv_files(self, comparison_df):
        """Generate CSV files for manual and ML results"""
        print("\n💾 GENERATING CSV FILES")
        print("=" * 60)
        
        # 1. Manual weights results CSV
        manual_csv = self.manual_results[['user_id', 'manual_financial_health_score', 'manual_engagement_score',
                                        'manual_financial_category', 'manual_enhanced_segment',
                                        'credit_score', 'dti_ratio', 'income', 'missed_payments',
                                        'click_rate', 'avg_time_viewed', 'total_interactions']].copy()
        manual_csv.columns = ['user_id', 'financial_health_score', 'engagement_score', 'financial_category', 
                             'enhanced_segment', 'credit_score', 'dti_ratio', 'income', 'missed_payments',
                             'click_rate', 'avg_time_viewed', 'total_interactions']
        manual_csv['weight_method'] = 'Manual'
        manual_csv.to_csv('manual_weights_results.csv', index=False)
        
        # 2. ML weights results CSV
        ml_csv = self.ml_results[['user_id', 'ml_financial_health_score', 'ml_engagement_score',
                                'ml_financial_category', 'ml_enhanced_segment',
                                'credit_score', 'dti_ratio', 'income', 'missed_payments',
                                'click_rate', 'avg_time_viewed', 'total_interactions']].copy()
        ml_csv.columns = ['user_id', 'financial_health_score', 'engagement_score', 'financial_category', 
                         'enhanced_segment', 'credit_score', 'dti_ratio', 'income', 'missed_payments',
                         'click_rate', 'avg_time_viewed', 'total_interactions']
        ml_csv['weight_method'] = 'ML_Learned'
        ml_csv.to_csv('ml_weights_results.csv', index=False)
        
        # 3. Combined comparison CSV
        comparison_csv = comparison_df.copy()
        comparison_csv.to_csv('manual_vs_ml_comparison.csv', index=False)
        
        print("✅ Generated CSV files:")
        print("   📄 manual_weights_results.csv - Results using manual weights")
        print("   📄 ml_weights_results.csv - Results using ML-learned weights") 
        print("   📄 manual_vs_ml_comparison.csv - Side-by-side comparison")
        
        # Print summary statistics
        print(f"\n📊 CSV FILE CONTENTS:")
        print(f"   Manual weights: {len(manual_csv)} users")
        print(f"   ML weights: {len(ml_csv)} users")
        print(f"   Comparison: {len(comparison_csv)} users")
        
        return manual_csv, ml_csv, comparison_csv
    
    def run_complete_analysis(self):
        """Run the complete manual vs ML comparison analysis"""
        print("🚀 STARTING COMPREHENSIVE MANUAL VS ML ANALYSIS")
        print("=" * 80)
        
        # Step 1: Calculate manual weights analysis
        manual_results = self.calculate_manual_weights_analysis()
        
        # Step 2: Run ML analysis
        ml_results = self.run_ml_analysis()
        
        # Step 3: Create comparison visualizations
        comparison_df = self.create_comparison_visualizations()
        
        # Step 4: Create detailed segment analysis
        self.create_detailed_segment_analysis(comparison_df)
        
        # Step 5: Generate CSV files
        manual_csv, ml_csv, comparison_csv = self.generate_csv_files(comparison_df)
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 80)
        print("✅ Manual weight analysis completed")
        print("✅ ML weight optimization completed")
        print("✅ Comprehensive visualizations created")
        print("✅ Detailed segment analysis created")
        print("✅ CSV files generated")
        print("\n📁 FILES CREATED:")
        print("   🖼️ manual_vs_ml_comprehensive_comparison.png")
        print("   🖼️ detailed_segment_analysis.png")
        print("   📄 manual_weights_results.csv")
        print("   📄 ml_weights_results.csv") 
        print("   📄 manual_vs_ml_comparison.csv")
        
        return {
            'manual_results': manual_results,
            'ml_results': ml_results,
            'comparison_df': comparison_df,
            'csv_files': {
                'manual': manual_csv,
                'ml': ml_csv,
                'comparison': comparison_csv
            }
        }

if __name__ == "__main__":
    print("🔬 MANUAL VS ML COMPREHENSIVE COMPARISON")
    print("=" * 80)
    
    try:
        # Create comparison analyzer
        analyzer = ManualVsMLComparison('joined_user_table.csv')
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        print("\n💡 KEY INSIGHTS TO LOOK FOR:")
        print("─" * 50)
        print("• How different are the financial category distributions?")
        print("• Which users changed segments between manual and ML?")
        print("• Do the ML weights reveal hidden patterns?")
        print("• Are the engagement score correlations high or low?")
        print("• Check the transition matrix to see category migrations!")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Please ensure the data file exists in the current directory")
    except Exception as e:
        print(f"❌ Error: {e}") 