import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SimpleClusterAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.user_features = None
        self.scaler = StandardScaler()
        
    def create_user_features(self):
        """Create user features for segmentation"""
        print("Creating user features...")
        
        # Basic engagement metrics by user
        user_stats = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).round(2)
        
        # Flatten column names
        user_stats.columns = ['avg_time_viewed', 'total_time_viewed', 'total_interactions',
                             'click_rate', 'total_clicks', 'unique_content_viewed']
        
        # Content type preferences
        content_preferences = self.df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
        content_preferences = content_preferences.div(content_preferences.sum(axis=1), axis=0)
        content_preferences.columns = [f'pref_{col}' for col in content_preferences.columns]
        
        # Combine features
        self.user_features = pd.concat([user_stats, content_preferences], axis=1).fillna(0)
        
        print(f"Created {len(self.user_features.columns)} features for {len(self.user_features)} users")
        return self.user_features
    
    def perform_clustering(self, n_clusters=3):
        """Perform K-means clustering"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Prepare features for clustering
        features_for_clustering = self.user_features.select_dtypes(include=[np.number]).fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_for_clustering)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.user_features['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Calculate individual engagement scores
        individual_scores = []
        for _, user_data in self.user_features.iterrows():
            score = (user_data['click_rate'] * 0.5 + 
                    min(user_data['avg_time_viewed'] / 60, 1) * 0.25 + 
                    min(user_data['total_interactions'] / 12, 1) * 0.25)
            individual_scores.append(score)
        
        self.user_features['individual_engagement_score'] = individual_scores
        
        # Define thresholds for individual engagement classification
        low_threshold = np.percentile(individual_scores, 33.33)
        high_threshold = np.percentile(individual_scores, 66.67)
        
        # Classify each user by individual engagement level
        def classify_individual_engagement(score):
            if score < low_threshold:
                return "Low"
            elif score < high_threshold:
                return "Medium"
            else:
                return "High"
        
        self.user_features['individual_engagement_level'] = [
            classify_individual_engagement(score) for score in individual_scores
        ]
        
        return self.user_features
    
    def analyze_cluster_composition(self):
        """Show detailed cluster composition"""
        print("\n" + "=" * 60)
        print("CLUSTER COMPOSITION ANALYSIS")
        print("=" * 60)
        
        # Show thresholds
        low_threshold = np.percentile(self.user_features['individual_engagement_score'], 33.33)
        high_threshold = np.percentile(self.user_features['individual_engagement_score'], 66.67)
        
        print(f"Individual Engagement Score Thresholds:")
        print(f"  Low Engagement: < {low_threshold:.3f}")
        print(f"  Medium Engagement: {low_threshold:.3f} - {high_threshold:.3f}")
        print(f"  High Engagement: > {high_threshold:.3f}")
        
        # Create cross-tabulation
        print(f"\nCLUSTER COMPOSITION BY INDIVIDUAL ENGAGEMENT LEVELS:")
        print("=" * 50)
        
        composition = pd.crosstab(
            self.user_features['cluster'], 
            self.user_features['individual_engagement_level'], 
            margins=True
        )
        print(composition)
        
        # Show percentages within each cluster
        print(f"\nPERCENTAGE BREAKDOWN WITHIN EACH CLUSTER:")
        print("=" * 50)
        composition_pct = pd.crosstab(
            self.user_features['cluster'], 
            self.user_features['individual_engagement_level'], 
            normalize='index'
        ) * 100
        print(composition_pct.round(1))
        
        # Detailed breakdown for each cluster
        print(f"\nDETAILED CLUSTER BREAKDOWN:")
        print("=" * 50)
        
        for cluster in sorted(self.user_features['cluster'].unique()):
            cluster_data = self.user_features[self.user_features['cluster'] == cluster]
            total_users = len(cluster_data)
            
            print(f"\nCluster {cluster} - {total_users} users:")
            
            engagement_breakdown = cluster_data['individual_engagement_level'].value_counts()
            for level in ['High', 'Medium', 'Low']:
                count = engagement_breakdown.get(level, 0)
                percentage = (count / total_users) * 100
                print(f"  {level} Individual Engagement: {count} users ({percentage:.1f}%)")
            
            # Show average scores
            avg_cluster_score = cluster_data['individual_engagement_score'].mean()
            print(f"  Average Individual Score: {avg_cluster_score:.3f}")
            print(f"  Score Range: {cluster_data['individual_engagement_score'].min():.3f} - {cluster_data['individual_engagement_score'].max():.3f}")
        
        return composition, composition_pct

# Run the analysis
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SimpleClusterAnalyzer('user_engagement_final.csv')
    
    # Run analysis
    analyzer.create_user_features()
    analyzer.perform_clustering()
    composition, composition_pct = analyzer.analyze_cluster_composition()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("This shows you how K-means clusters (0, 1, 2) are composed")
    print("of users with different individual engagement levels (High, Medium, Low)") 