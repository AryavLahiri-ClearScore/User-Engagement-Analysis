import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinanciallyAwareRecommender:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.user_features = None
        self.scaler = StandardScaler()
        
    def create_financial_health_score(self):
        """Create a comprehensive financial health score with absolute threshold categorization"""
        print("Creating financial health scores with absolute categorization thresholds...")
        
        # Get unique users with their financial attributes
        user_financial = self.df.groupby('user_id').first()[
            ['total_debt', 'credit_score', 'missed_payments', 
             'has_mortgage', 'has_car', 'has_ccj', 'dti_ratio', 'income']
        ]
        
        financial_scores = []
        
        for _, user in user_financial.iterrows():
            # Credit Score Component (0-1, higher is better)
            credit_component = min(user['credit_score'] / 1000, 1.0)
            
            # Debt-to-Income Ratio Component (0-1, lower is better)
            dti_component = max(0, 1 - user['dti_ratio'])
            
            # Missed Payments Component (0-1, fewer is better)
            missed_payments_component = max(0, 1 - (user['missed_payments'] / 10))
            
            # Income Component (normalized, higher is better)
            income_component = min(user['income'] / 100000, 1.0)
            
            # CCJ Component (0-1, no CCJ is better)
            ccj_component = 0.0 if user['has_ccj'] else 1.0
            
            # Asset Component (having mortgage/car indicates stability)
            asset_component = (user['has_mortgage'] * 0.6 + user['has_car'] * 0.4)
            
            # Calculate composite financial health score (0-1) with improved weighting
            financial_health = (
                credit_component * 0.30 +
                dti_component * 0.25 +
                missed_payments_component * 0.15 +
                income_component * 0.15 +
                ccj_component * 0.10 +
                asset_component * 0.05
            )
            
            financial_scores.append(financial_health)
        
        user_financial['financial_health_score'] = financial_scores
        
        # ABSOLUTE THRESHOLDS based on real UK financial health standards
        def categorize_financial_health(score):
            if score >= 0.8:
                return "Excellent"      # Strong across all metrics
            elif score >= 0.65:
                return "Good"           # Above average financial health
            elif score >= 0.45:
                return "Fair"           # Some concerns but manageable
            else:
                return "Poor"           # Significant financial challenges
        
        user_financial['financial_category'] = user_financial['financial_health_score'].apply(categorize_financial_health)
        
        print("Financial health distribution with absolute thresholds:")
        print(user_financial['financial_category'].value_counts())
        print(f"Score ranges: {user_financial['financial_health_score'].min():.3f} - {user_financial['financial_health_score'].max():.3f}")
        
        # Show categorization thresholds used
        print("\nAbsolute Categorization Thresholds Applied:")
        print("Excellent: Score >= 0.8 (Strong financial health across all metrics)")
        print("Good: Score >= 0.65 (Above average financial health)")
        print("Fair: Score >= 0.45 (Some concerns but manageable)")
        print("Poor: Score < 0.45 (Significant financial challenges)")
        
        return user_financial
    
    def create_enhanced_user_features(self):
        """Create user features including both engagement and financial data"""
        print("Creating enhanced user features...")
        
        # Basic engagement metrics
        user_stats = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).round(2)
        
        user_stats.columns = ['avg_time_viewed', 'total_time_viewed', 'total_interactions',
                             'click_rate', 'total_clicks', 'unique_content_viewed']
        
        # Content type preferences
        content_preferences = self.df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
        content_preferences = content_preferences.div(content_preferences.sum(axis=1), axis=0)
        content_preferences.columns = [f'pref_{col}' for col in content_preferences.columns]
        
        # Financial health scores
        financial_data = self.create_financial_health_score()
        
        # Combine all features
        self.user_features = pd.concat([
            user_stats,
            content_preferences,
            financial_data
        ], axis=1).fillna(0)
        
        print(f"Created {len(self.user_features.columns)} features for {len(self.user_features)} users")
        return self.user_features
    
    def perform_enhanced_segmentation(self):
        """Perform segmentation including financial health"""
        print("Performing enhanced segmentation...")
        
        # Select features for clustering (engagement + financial)
        engagement_features = ['avg_time_viewed', 'total_interactions', 'click_rate', 'unique_content_viewed']
        financial_features = ['financial_health_score', 'credit_score', 'dti_ratio', 'income']
        content_features = [col for col in self.user_features.columns if col.startswith('pref_')]
        
        clustering_features = engagement_features + financial_features + content_features
        features_for_clustering = self.user_features[clustering_features].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_for_clustering)
        
        # Perform K-means clustering with more clusters for nuanced segmentation
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        self.user_features['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Create composite engagement-financial score
        engagement_scores = []
        for _, user_data in self.user_features.iterrows():
            engagement_score = (
                user_data['click_rate'] * 0.4 + 
                min(user_data['avg_time_viewed'] / 60, 1) * 0.3 + 
                min(user_data['total_interactions'] / 12, 1) * 0.3
            )
            engagement_scores.append(engagement_score)
        
        self.user_features['engagement_score'] = engagement_scores
        
        # Assign meaningful segment names based on engagement and financial health
        def assign_segment_name(row):
            engagement = row['engagement_score']
            financial_cat = row['financial_category']
            
            if engagement > 0.5:
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
        
        self.user_features['enhanced_segment'] = self.user_features.apply(assign_segment_name, axis=1)
        
        print("Enhanced segment distribution:")
        print(self.user_features['enhanced_segment'].value_counts())
        
        return self.user_features
    
    def create_financial_visualizations(self):
        """Create comprehensive financial and engagement visualizations"""
        print("\n" + "=" * 60)
        print("CREATING FINANCIAL ANALYSIS VISUALaIZATIONS")
        print("=" * 60)
        
        # FIGURE 1: Financial Health Dashboard (2x3 grid)
        fig1, axes1 = plt.subplots(2, 3, figsize=(20, 12))
        fig1.suptitle('Financial Health & Engagement Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Enhanced segment distribution
        segment_counts = self.user_features['enhanced_segment'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        axes1[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes1[0, 0].set_title('Enhanced Segment Distribution')
        
        # 2. Financial category distribution
        financial_counts = self.user_features['financial_category'].value_counts()
        colors_fin = ['#ff9999', '#66b3ff', '#99ff99']
        axes1[0, 1].bar(financial_counts.index, financial_counts.values, color=colors_fin)
        axes1[0, 1].set_title('Financial Health Categories')
        axes1[0, 1].set_ylabel('Number of Users')
        axes1[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Credit score distribution
        axes1[0, 2].hist(self.user_features['credit_score'], bins=20, alpha=0.7, 
                        color='skyblue', edgecolor='black')
        axes1[0, 2].axvline(self.user_features['credit_score'].mean(), color='red', 
                           linestyle='--', label=f'Mean: {self.user_features["credit_score"].mean():.0f}')
        axes1[0, 2].set_title('Credit Score Distribution')
        axes1[0, 2].set_xlabel('Credit Score')
        axes1[0, 2].set_ylabel('Number of Users')
        axes1[0, 2].legend()
        
        # 4. DTI Ratio vs Financial Health Score
        scatter = axes1[1, 0].scatter(self.user_features['dti_ratio'], 
                                     self.user_features['financial_health_score'],
                                     c=self.user_features['credit_score'], 
                                     cmap='viridis', alpha=0.7, s=50)
        axes1[1, 0].set_xlabel('Debt-to-Income Ratio')
        axes1[1, 0].set_ylabel('Financial Health Score')
        axes1[1, 0].set_title('DTI vs Financial Health (colored by Credit Score)')
        plt.colorbar(scatter, ax=axes1[1, 0], label='Credit Score')
        
        # 5. Engagement vs Financial Health by segment
        for segment in self.user_features['enhanced_segment'].unique():
            segment_data = self.user_features[self.user_features['enhanced_segment'] == segment]
            axes1[1, 1].scatter(segment_data['engagement_score'], 
                               segment_data['financial_health_score'],
                               label=segment, alpha=0.7, s=50)
        axes1[1, 1].set_xlabel('Engagement Score')
        axes1[1, 1].set_ylabel('Financial Health Score')
        axes1[1, 1].set_title('Engagement vs Financial Health by Segment')
        axes1[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 6. Income distribution by financial category
        financial_categories = self.user_features['financial_category'].unique()
        income_data = [self.user_features[self.user_features['financial_category'] == cat]['income'].values 
                      for cat in financial_categories]
        box_plot = axes1[1, 2].boxplot(income_data, labels=financial_categories, patch_artist=True)
        colors_box = ['#ff9999', '#66b3ff', '#99ff99']
        for patch, color in zip(box_plot['boxes'], colors_box):
            patch.set_facecolor(color)
        axes1[1, 2].set_title('Income Distribution by Financial Category')
        axes1[1, 2].set_ylabel('Income (£)')
        axes1[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('financial_health_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: financial_health_dashboard.png")
        
        # FIGURE 2: Financial Metrics Deep Dive (2x2 grid)
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
        fig2.suptitle('Financial Metrics Deep Dive', fontsize=16, fontweight='bold')
        
        # 1. Financial health score distribution
        axes2[0, 0].hist(self.user_features['financial_health_score'], bins=20, 
                        alpha=0.7, color='green', edgecolor='black')
        axes2[0, 0].axvline(self.user_features['financial_health_score'].mean(), 
                           color='red', linestyle='--', 
                           label=f'Mean: {self.user_features["financial_health_score"].mean():.2f}')
        axes2[0, 0].set_title('Financial Health Score Distribution')
        axes2[0, 0].set_xlabel('Financial Health Score')
        axes2[0, 0].set_ylabel('Number of Users')
        axes2[0, 0].legend()
        
        # 2. Missed payments analysis
        missed_payments_dist = self.user_features['missed_payments'].value_counts().sort_index()
        axes2[0, 1].bar(missed_payments_dist.index, missed_payments_dist.values, 
                       color='coral', alpha=0.7)
        axes2[0, 1].set_title('Missed Payments Distribution')
        axes2[0, 1].set_xlabel('Number of Missed Payments')
        axes2[0, 1].set_ylabel('Number of Users')
        
        # 3. Asset ownership analysis
        asset_data = pd.DataFrame({
            'Has Mortgage': self.user_features['has_mortgage'].sum(),
            'Has Car': self.user_features['has_car'].sum(),
            'Has CCJ': self.user_features['has_ccj'].sum()
        }, index=[0])
        asset_data.T.plot(kind='bar', ax=axes2[1, 0], color=['green', 'blue', 'red'], 
                         alpha=0.7, legend=False)
        axes2[1, 0].set_title('Asset Ownership & Financial Issues')
        axes2[1, 0].set_ylabel('Number of Users')
        axes2[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Financial metrics correlation heatmap
        financial_metrics = ['credit_score', 'dti_ratio', 'income', 'total_debt', 
                           'missed_payments', 'financial_health_score']
        correlation_matrix = self.user_features[financial_metrics].corr()
        
        im = axes2[1, 1].imshow(correlation_matrix.values, cmap='RdBu_r', 
                               aspect='auto', vmin=-1, vmax=1)
        axes2[1, 1].set_xticks(range(len(correlation_matrix.columns)))
        axes2[1, 1].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes2[1, 1].set_yticks(range(len(correlation_matrix.index)))
        axes2[1, 1].set_yticklabels(correlation_matrix.index)
        axes2[1, 1].set_title('Financial Metrics Correlation')
        
        # Add correlation values to heatmap
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text_color = 'white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black'
                axes2[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                                ha='center', va='center', color=text_color)
        
        plt.colorbar(im, ax=axes2[1, 1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig('financial_metrics_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: financial_metrics_analysis.png")
        
        # FIGURE 3: Segment Analysis & Recommendations (2x2 grid)
        fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))
        fig3.suptitle('Enhanced Segment Analysis & Content Recommendations', fontsize=16, fontweight='bold')
        
        # 1. Average financial health by segment
        segment_financial_health = self.user_features.groupby('enhanced_segment')['financial_health_score'].mean().sort_values(ascending=False)
        axes3[0, 0].bar(range(len(segment_financial_health)), segment_financial_health.values, 
                       color='lightgreen', alpha=0.7)
        axes3[0, 0].set_xticks(range(len(segment_financial_health)))
        axes3[0, 0].set_xticklabels(segment_financial_health.index, rotation=45, ha='right')
        axes3[0, 0].set_title('Average Financial Health by Segment')
        axes3[0, 0].set_ylabel('Financial Health Score')
        
        # 2. Credit score by segment (box plot)
        segments = self.user_features['enhanced_segment'].unique()
        credit_data = [self.user_features[self.user_features['enhanced_segment'] == seg]['credit_score'].values 
                      for seg in segments]
        box_plot2 = axes3[0, 1].boxplot(credit_data, labels=segments, patch_artist=True)
        colors_segments = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        for patch, color in zip(box_plot2['boxes'], colors_segments):
            patch.set_facecolor(color)
        axes3[0, 1].set_title('Credit Score Distribution by Segment')
        axes3[0, 1].set_ylabel('Credit Score')
        axes3[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Content preferences heatmap by enhanced segment
        content_prefs = self.user_features.groupby('enhanced_segment')[[col for col in self.user_features.columns if col.startswith('pref_')]].mean()
        content_prefs.columns = [col.replace('pref_', '') for col in content_prefs.columns]
        
        im2 = axes3[1, 0].imshow(content_prefs.values, cmap='YlOrRd', aspect='auto')
        axes3[1, 0].set_xticks(range(len(content_prefs.columns)))
        axes3[1, 0].set_xticklabels(content_prefs.columns, rotation=45)
        axes3[1, 0].set_yticks(range(len(content_prefs.index)))
        axes3[1, 0].set_yticklabels(content_prefs.index, fontsize=8)
        axes3[1, 0].set_title('Content Preferences by Enhanced Segment')
        plt.colorbar(im2, ax=axes3[1, 0], fraction=0.046, pad=0.04)
        
        # 4. Financial vs Engagement score comparison
        financial_categories = ['Poor', 'Fair', 'Good', 'Excellent']
        fin_eng_comparison = self.user_features.groupby('financial_category')[['financial_health_score', 'engagement_score']].mean()
        
        x = np.arange(len(financial_categories))
        width = 0.35
        
        bars1 = axes3[1, 1].bar(x - width/2, fin_eng_comparison['financial_health_score'], 
                               width, label='Financial Health', color='lightblue', alpha=0.7)
        bars2 = axes3[1, 1].bar(x + width/2, fin_eng_comparison['engagement_score'], 
                               width, label='Engagement Score', color='lightcoral', alpha=0.7)
        
        axes3[1, 1].set_xlabel('Financial Category')
        axes3[1, 1].set_ylabel('Score')
        axes3[1, 1].set_title('Financial Health vs Engagement by Category')
        axes3[1, 1].set_xticks(x)
        axes3[1, 1].set_xticklabels(financial_categories)
        axes3[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('enhanced_engagement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: enhanced_engagement_analysis.png")
    
    def visualize_financial_clustering(self):
        """Visualize the enhanced clustering with financial context"""
        print("\n" + "=" * 60)
        print("FINANCIAL CLUSTERING VISUALIZATION")
        print("=" * 60)
        
        # Prepare features for clustering (same as used in segmentation)
        engagement_features = ['avg_time_viewed', 'total_interactions', 'click_rate', 'unique_content_viewed']
        financial_features = ['financial_health_score', 'credit_score', 'dti_ratio', 'income']
        content_features = [col for col in self.user_features.columns if col.startswith('pref_')]
        
        clustering_features = engagement_features + financial_features + content_features
        features_for_clustering = self.user_features[clustering_features].fillna(0)
        features_scaled = self.scaler.fit_transform(features_for_clustering)
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create the visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Financial-Engagement Clustering', fontsize=16, fontweight='bold')
        
        # 1. Clusters by enhanced segment
        segment_colors = plt.cm.Set3(np.linspace(0, 1, len(self.user_features['enhanced_segment'].unique())))
        segments = self.user_features['enhanced_segment'].unique()
        
        for i, segment in enumerate(segments):
            mask = self.user_features['enhanced_segment'] == segment
            if mask.any():
                ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[segment_colors[i]], label=segment, alpha=0.7, s=50)
        ax1.set_title('Clusters by Enhanced Segment')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 2. Colored by financial health score
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=self.user_features['financial_health_score'], 
                              cmap='RdYlGn', alpha=0.7, s=50)
        ax2.set_title('Colored by Financial Health Score')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter2, ax=ax2, label='Financial Health Score')
        
        # 3. Colored by credit score
        scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=self.user_features['credit_score'], 
                              cmap='viridis', alpha=0.7, s=50)
        ax3.set_title('Colored by Credit Score')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter3, ax=ax3, label='Credit Score')
        
        # 4. Colored by DTI ratio
        scatter4 = ax4.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=self.user_features['dti_ratio'], 
                              cmap='Reds', alpha=0.7, s=50)
        ax4.set_title('Colored by Debt-to-Income Ratio')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter4, ax=ax4, label='DTI Ratio')
        
        plt.tight_layout()
        plt.savefig('financial_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: financial_clustering_analysis.png")
        
        # Print PCA explanation
        print(f"PCA Explained Variance:")
        print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"  Total: {sum(pca.explained_variance_ratio_):.1%}")
        
        return features_2d, pca
    
    def generate_financial_content_recommendations(self):
        """Generate financially-aware content recommendations"""
        print("Generating financially-aware recommendations...")
        
        recommendations = []
        content_types = ['improve', 'insights', 'drivescore', 'protect', 'credit_cards', 'loans']
        
        for user_id, user_data in self.user_features.iterrows():
            segment = user_data['enhanced_segment']
            financial_cat = user_data['financial_category']
            credit_score = user_data['credit_score']
            dti_ratio = user_data['dti_ratio']
            has_ccj = user_data['has_ccj']
            missed_payments = user_data['missed_payments']
            
            # Financial priority recommendations
            financial_priorities = self.get_financial_priorities(user_data)
            
            # Base content preferences
            user_prefs = {}
            for content_type in content_types:
                user_prefs[content_type] = user_data.get(f'pref_{content_type}', 0)
            
            # Apply financial context to content scoring
            content_scores = {}
            
            for content_type in content_types:
                base_score = user_prefs[content_type]
                
                # Apply financial relevance multipliers
                if content_type == 'improve' and credit_score < 650:
                    base_score *= 2.0  # Credit improvement is high priority
                elif content_type == 'protect' and financial_cat == "Excellent":
                    base_score *= 1.5  # Wealth protection for financially healthy users
                elif content_type == 'loans' and dti_ratio > 0.6:
                    base_score *= 0.5  # Reduce loan recommendations for high DTI
                elif content_type == 'credit_cards' and has_ccj:
                    base_score *= 0.3  # Reduce credit card recommendations for CCJ users
                elif content_type == 'drivescore' and financial_cat == "Poor":
                    base_score *= 1.8  # Financial education priority for financially challenged users
                elif content_type == 'insights' and missed_payments > 2:
                    base_score *= 1.7  # Financial insights for users with payment issues
                
                content_scores[content_type] = base_score
            
            # Sort by adjusted scores
            sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Generate segment-specific strategies
            strategy = self.get_strategy_by_segment(segment, user_data)
            
            # Add urgency flags
            urgency_flags = self.get_urgency_flags(user_data)
            
            recommendations.append({
                'user_id': user_id,
                'enhanced_segment': segment,
                'financial_category': financial_cat,
                'primary_recommendation': sorted_content[0][0],
                'secondary_recommendation': sorted_content[1][0],
                'tertiary_recommendation': sorted_content[2][0],
                'financial_priorities': ', '.join(financial_priorities),
                'urgency_flags': ', '.join(urgency_flags),
                'strategy': strategy,
                'credit_score': credit_score,
                'dti_ratio': dti_ratio,
                'financial_health_score': user_data['financial_health_score'],
                'engagement_score': user_data['engagement_score']
            })
        
        return pd.DataFrame(recommendations)
    
    def get_financial_priorities(self, user_data):
        """Identify financial priorities for a user"""
        priorities = []
        
        if user_data['credit_score'] < 650:
            priorities.append("Credit_Repair")
        if user_data['dti_ratio'] > 0.6:
            priorities.append("Debt_Reduction")
        if user_data['missed_payments'] > 2:
            priorities.append("Payment_Management")
        if user_data['has_ccj']:
            priorities.append("Legal_Financial_Issues")
        if user_data['total_debt'] > user_data['income'] * 0.8:
            priorities.append("Debt_Consolidation")
        if user_data['financial_health_score'] > 0.7 and user_data['income'] > 50000:
            priorities.append("Wealth_Building")
        if not user_data['has_mortgage'] and user_data['financial_health_score'] > 0.6:
            priorities.append("Homeownership_Ready")
        
        return priorities if priorities else ["General_Financial_Wellness"]
    
    def get_urgency_flags(self, user_data):
        """Identify urgent financial issues"""
        flags = []
        
        if user_data['dti_ratio'] > 0.8:
            flags.append("HIGH_DEBT_BURDEN")
        if user_data['credit_score'] < 500:
            flags.append("CRITICAL_CREDIT_SCORE")
        if user_data['missed_payments'] > 4:
            flags.append("PAYMENT_CRISIS")
        if user_data['has_ccj']:
            flags.append("LEGAL_ACTION")
        
        return flags if flags else ["STABLE_FINANCIAL_POSITION"]
    
    def get_strategy_by_segment(self, segment, user_data):
        """Get tailored strategy based on enhanced segment"""
        strategies = {
            "Premium_Engaged": f"Offer premium wealth management and investment content. Focus on portfolio optimization and advanced financial strategies. Credit score: {user_data['credit_score']}.",
            
            "Growth_Focused": f"Provide growth-oriented financial content with moderate complexity. Focus on building wealth and improving financial position. Current DTI: {user_data['dti_ratio']:.2f}.",
            
            "Recovery_Engaged": f"Deliver financial recovery content with high engagement. Focus on debt management and credit repair while maintaining engagement. Priority: Credit improvement from {user_data['credit_score']}.",
            
            "Premium_Moderate": f"Offer premium content with clear value propositions. Balance wealth building with practical financial advice. Leverage high financial health score: {user_data['financial_health_score']:.2f}.",
            
            "Mainstream": f"Provide balanced financial content for users with decent financial health. Focus on practical advice and gradual improvement. Build on solid financial foundation.",
            
            "Recovery_Moderate": f"Deliver accessible financial recovery content. Simplify complex concepts and focus on immediate actionable steps. Address DTI ratio: {user_data['dti_ratio']:.2f}.",
            
            "Financial_Priority": f"Urgent: Focus on critical financial issues first. Provide crisis management content and immediate help resources. Address multiple risk factors.",
            
            "Activation_Needed": f"Basic financial education and engagement building. Start with simple concepts and gradually increase complexity. Build financial awareness."
        }
        
        return strategies.get(segment, "Provide general financial guidance based on user profile.")
    
    def print_enhanced_recommendations(self, recommendations_df, n_samples=10):
        """Print sample enhanced recommendations"""
        print("\n" + "=" * 80)
        print("ENHANCED FINANCIALLY-AWARE RECOMMENDATIONS")
        print("=" * 80)
        
        for _, user in recommendations_df.head(n_samples).iterrows():
            print(f"\nUser: {user['user_id']}")
            print(f"Enhanced Segment: {user['enhanced_segment']}")
            print(f"Financial Category: {user['financial_category']}")
            print(f"Credit Score: {user['credit_score']} | DTI: {user['dti_ratio']:.2f} | Health Score: {user['financial_health_score']:.2f}")
            print(f"Primary Rec: {user['primary_recommendation']}")
            print(f"Financial Priorities: {user['financial_priorities']}")
            print(f"Urgency Flags: {user['urgency_flags']}")
            print(f"Strategy: {user['strategy'][:100]}...")
            print("-" * 80)
    
    def analyze_segments_by_financial_health(self):
        """Analyze the relationship between segments and financial health"""
        print("\n" + "=" * 60)
        print("SEGMENT ANALYSIS BY FINANCIAL HEALTH")
        print("=" * 60)
        
        segment_analysis = self.user_features.groupby('enhanced_segment').agg({
            'financial_health_score': ['mean', 'std'],
            'credit_score': ['mean', 'min', 'max'],
            'dti_ratio': ['mean', 'std'],
            'income': ['mean', 'median'],
            'engagement_score': ['mean']
        }).round(2)
        
        print(segment_analysis)
        
        return segment_analysis
    
    def run_enhanced_analysis(self):
        """Run the complete enhanced analysis"""
        print("Starting Enhanced Financially-Aware Recommendation Analysis...")
        
        # Create enhanced features
        self.create_enhanced_user_features()
        
        # Perform enhanced segmentation
        self.perform_enhanced_segmentation()
        
        # Generate recommendations
        recommendations = self.generate_financial_content_recommendations()
        
        # Create comprehensive visualizations
        self.create_financial_visualizations()
        
        # Create clustering visualizations
        self.visualize_financial_clustering()
        
        # Print sample recommendations
        self.print_enhanced_recommendations(recommendations)
        
        # Analyze segments
        self.analyze_segments_by_financial_health()
        
        return recommendations

if __name__ == "__main__":
    # Initialize the enhanced recommender
    recommender = FinanciallyAwareRecommender('joined_user_table.csv')
    
    # Run enhanced analysis
    enhanced_recommendations = recommender.run_enhanced_analysis()
    
    # Save enhanced recommendations
    enhanced_recommendations.to_csv('enhanced_financial_recommendations.csv', index=False)
    print(f"\nEnhanced recommendations saved to 'enhanced_financial_recommendations.csv'")
    
    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("=" * 60) 