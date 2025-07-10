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

class UserEngagementAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.user_features = None
        self.segments = None
        self.scaler = StandardScaler()
        
    def load_and_explore_data(self):
        """Load and explore the engagement data"""
        print("=" * 60)
        print("USER ENGAGEMENT DATA ANALYSIS")
        print("=" * 60)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Number of unique users: {self.df['user_id'].nunique()}")
        print(f"Number of unique content pieces: {self.df['content_id'].nunique()}")
        
        print("\nContent types distribution:")
        print(self.df['content_type'].value_counts())
        
        print(f"\nOverall click rate: {self.df['clicked'].mean():.2%}")
        print(f"Average time viewed: {self.df['time_viewed_in_sec'].mean():.1f} seconds")
        
        return self.df.describe()
    
    def create_user_features(self):
        """Create comprehensive user features for segmentation"""
        print("\n" + "=" * 60)
        print("CREATING USER FEATURES")
        print("=" * 60)
        
        # Basic engagement metrics by user
        user_stats = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).round(2)
        
        # Flatten column names
        user_stats.columns = ['avg_time_viewed', 'total_time_viewed', 'total_interactions',
                             'click_rate', 'total_clicks', 'unique_content_viewed']
        
        # Content type preferences (percentage of interactions per content type)
        content_preferences = self.df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
        content_preferences = content_preferences.div(content_preferences.sum(axis=1), axis=0)
        content_preferences.columns = [f'pref_{col}' for col in content_preferences.columns]
        
        # Average time spent per content type
        avg_time_by_content = self.df.groupby(['user_id', 'content_type'])['time_viewed_in_sec'].mean().unstack(fill_value=0)
        avg_time_by_content.columns = [f'avg_time_{col}' for col in avg_time_by_content.columns]
        
        # Click rates by content type
        click_rates_by_content = self.df.groupby(['user_id', 'content_type'])['clicked'].mean().unstack(fill_value=0)
        click_rates_by_content.columns = [f'click_rate_{col}' for col in click_rates_by_content.columns]
        
        # Engagement patterns
        engagement_patterns = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['std', 'min', 'max'],
        }).round(2)
        engagement_patterns.columns = ['time_std', 'min_time', 'max_time']
        
        # Combine all features
        self.user_features = pd.concat([
            user_stats,
            content_preferences,
            avg_time_by_content,
            click_rates_by_content,
            engagement_patterns
        ], axis=1).fillna(0)
        
        print(f"Created {len(self.user_features.columns)} features for {len(self.user_features)} users")
        return self.user_features
    
    def perform_user_segmentation(self, n_clusters=4):
        """Perform user segmentation using K-means clustering"""
        print("\n" + "=" * 60)
        print("PERFORMING USER SEGMENTATION")
        print("=" * 60)
        
        # Prepare features for clustering
        features_for_clustering = self.user_features.select_dtypes(include=[np.number]).fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_for_clustering)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.user_features['segment'] = kmeans.fit_predict(features_scaled)
        
        # Analyze segments
        print(f"Created {n_clusters} user segments:")
        segment_summary = self.user_features.groupby('segment').agg({
            'total_interactions': 'mean',
            'avg_time_viewed': 'mean',
            'click_rate': 'mean',
            'total_time_viewed': 'mean',
            'unique_content_viewed': 'mean'
        }).round(2)
        
        print(segment_summary)
        
        # Name segments based on characteristics
        segment_names = self.name_segments(segment_summary)
        self.user_features['segment_name'] = self.user_features['segment'].map(segment_names)
        
        # Debug: Show actual user counts per segment
        print("\nDEBUG: Actual user counts per segment:")
        segment_counts = self.user_features['segment_name'].value_counts()
        print(segment_counts)
        
        return self.user_features
    
    def name_segments(self, segment_summary):
        """Name segments based on their characteristics"""
        segment_names = {}
        
        print("\nDEBUG: Segment characteristics before naming:")
        print(segment_summary)
        
        for segment in segment_summary.index:
            stats = segment_summary.loc[segment]
            
            print(f"\nSegment {segment}:")
            print(f"  Click rate: {stats['click_rate']:.3f}")
            print(f"  Avg time viewed: {stats['avg_time_viewed']:.1f}s")
            print(f"  Total interactions: {stats['total_interactions']:.1f}")
            
            if stats['click_rate'] > 0.5 and stats['avg_time_viewed'] > 30:
                segment_names[segment] = "Highly Engaged"
                print(f"  -> Assigned: Highly Engaged")
            elif stats['click_rate'] > 0.3 and stats['total_interactions'] > 8:
                segment_names[segment] = "Active Users"
                print(f"  -> Assigned: Active Users")
            elif stats['avg_time_viewed'] > 25 and stats['click_rate'] < 0.3:
                segment_names[segment] = "Browsers"
                print(f"  -> Assigned: Browsers")
            else:
                segment_names[segment] = "Casual Users"
                print(f"  -> Assigned: Casual Users")
        print("Segment names: ", segment_names)
        return segment_names
    
    def analyze_content_preferences(self):
        """Analyze content preferences by segment"""
        print("\n" + "=" * 60)
        print("CONTENT PREFERENCES BY SEGMENT")
        print("=" * 60)
        
        # Content type preferences by segment
        content_prefs = self.user_features.groupby('segment_name')[[col for col in self.user_features.columns if col.startswith('pref_')]].mean()
        content_prefs.columns = [col.replace('pref_', '') for col in content_prefs.columns]
        
        print("Average content preferences by segment:")
        print(content_prefs.round(3))
        
        # Average time spent on each content type by segment
        time_prefs = self.user_features.groupby('segment_name')[[col for col in self.user_features.columns if col.startswith('avg_time_')]].mean()
        time_prefs.columns = [col.replace('avg_time_', '') for col in time_prefs.columns]
        
        print("\nAverage time spent on content by segment:")
        print(time_prefs.round(1))
        
        # Click rates by content type and segment
        click_prefs = self.user_features.groupby('segment_name')[[col for col in self.user_features.columns if col.startswith('click_rate_')]].mean()
        click_prefs.columns = [col.replace('click_rate_', '') for col in click_prefs.columns]
        
        print("\nClick rates by content type and segment:")
        print(click_prefs.round(3))
        
        return content_prefs, time_prefs, click_prefs
    
    def generate_personalized_recommendations(self):
        """Generate personalized content recommendations for each user"""
        print("\n" + "=" * 60)
        print("GENERATING PERSONALIZED RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # Content types available
        content_types = ['improve', 'insights', 'drivescore', 'protect', 'credit_cards', 'loans']
        
        for user_id, user_data in self.user_features.iterrows():
            segment = user_data['segment_name']
            
            # Get user's content preferences
            user_prefs = {}
            user_avg_times = {}
            user_click_rates = {}
            
            for content_type in content_types:
                user_prefs[content_type] = user_data.get(f'pref_{content_type}', 0)
                user_avg_times[content_type] = user_data.get(f'avg_time_{content_type}', 0)
                user_click_rates[content_type] = user_data.get(f'click_rate_{content_type}', 0)
            
            # Calculate recommendation scores
            rec_scores = {}
            for content_type in content_types:
                # Combine preference, time spent, and click rate
                pref_score = user_prefs[content_type] * 0.3
                time_score = min(user_avg_times[content_type] / 60, 1) * 0.4  # Normalize by 60 seconds
                click_score = user_click_rates[content_type] * 0.3
                
                rec_scores[content_type] = pref_score + time_score + click_score
            
            # Sort by recommendation score
            sorted_content = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Generate recommendations based on segment
            if segment == "Highly Engaged":
                primary_rec = sorted_content[0][0]
                secondary_rec = sorted_content[1][0]
                strategy = "Provide in-depth, detailed content"
            elif segment == "Active Users":
                primary_rec = sorted_content[0][0]
                secondary_rec = sorted_content[1][0]
                strategy = "Offer varied content with clear value propositions"
            elif segment == "Browsers":
                primary_rec = sorted_content[0][0]
                secondary_rec = sorted_content[1][0]
                strategy = "Focus on visually engaging, easy-to-scan content"
            else:  # Casual Users
                primary_rec = sorted_content[0][0]
                secondary_rec = sorted_content[1][0]
                strategy = "Provide simple, bite-sized content with clear benefits"
            
            recommendations.append({
                'user_id': user_id,
                'segment': segment,
                'primary_recommendation': primary_rec,
                'secondary_recommendation': secondary_rec,
                'strategy': strategy,
                'user_click_rate': user_data['click_rate'],
                'user_avg_time': user_data['avg_time_viewed'],
                'total_interactions': user_data['total_interactions']
            })
        
        return pd.DataFrame(recommendations)
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('User Engagement Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Segment distribution
        segment_counts = self.user_features['segment_name'].value_counts()
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('User Segment Distribution')
        
        # 2. Content type popularity
        content_popularity = self.df['content_type'].value_counts()
        axes[0, 1].bar(content_popularity.index, content_popularity.values)
        axes[0, 1].set_title('Content Type Popularity')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Click rate by content type
        click_rates = self.df.groupby('content_type')['clicked'].mean().sort_values(ascending=False)
        axes[0, 2].bar(click_rates.index, click_rates.values)
        axes[0, 2].set_title('Click Rate by Content Type')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].set_ylabel('Click Rate')
        
        # 4. Average time viewed by segment
        avg_times = self.user_features.groupby('segment_name')['avg_time_viewed'].mean()
        axes[1, 0].bar(avg_times.index, avg_times.values)
        axes[1, 0].set_title('Average Time Viewed by Segment')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylabel('Seconds')
        
        # 5. Interactions vs Click Rate by segment
        for segment in self.user_features['segment_name'].unique():
            segment_data = self.user_features[self.user_features['segment_name'] == segment]
            axes[1, 1].scatter(segment_data['total_interactions'], segment_data['click_rate'], 
                             label=segment, alpha=0.7)
        axes[1, 1].set_xlabel('Total Interactions')
        axes[1, 1].set_ylabel('Click Rate')
        axes[1, 1].set_title('Interactions vs Click Rate by Segment')
        axes[1, 1].legend()
        
        # 6. Content preference heatmap
        content_prefs = self.user_features.groupby('segment_name')[[col for col in self.user_features.columns if col.startswith('pref_')]].mean()
        content_prefs.columns = [col.replace('pref_', '') for col in content_prefs.columns]
        
        im = axes[1, 2].imshow(content_prefs.values, cmap='YlOrRd', aspect='auto')
        axes[1, 2].set_xticks(range(len(content_prefs.columns)))
        axes[1, 2].set_xticklabels(content_prefs.columns, rotation=45)
        axes[1, 2].set_yticks(range(len(content_prefs.index)))
        axes[1, 2].set_yticklabels(content_prefs.index)
        axes[1, 2].set_title('Content Preferences by Segment')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    def print_sample_recommendations(self, recommendations_df, n_samples=10):
        """Print sample recommendations for users"""
        print("\n" + "=" * 60)
        print("SAMPLE PERSONALIZED RECOMMENDATIONS")
        print("=" * 60)
        
        sample_users = recommendations_df.head(n_samples)
        
        for _, user in sample_users.iterrows():
            print(f"\nUser: {user['user_id']}")
            print(f"Segment: {user['segment']}")
            print(f"Primary Recommendation: {user['primary_recommendation']}")
            print(f"Secondary Recommendation: {user['secondary_recommendation']}")
            print(f"Strategy: {user['strategy']}")
            print(f"User Stats: {user['total_interactions']} interactions, "
                  f"{user['user_click_rate']:.1%} click rate, "
                  f"{user['user_avg_time']:.1f}s avg time")
            print("-" * 50)
    
    def generate_segment_insights(self):
        """Generate detailed insights for each segment"""
        print("\n" + "=" * 60)
        print("SEGMENT INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)
        
        for segment in self.user_features['segment_name'].unique():
            segment_data = self.user_features[self.user_features['segment_name'] == segment]
            
            print(f"\n{segment.upper()} SEGMENT")
            print("=" * 40)
            print(f"Size: {len(segment_data)} users ({len(segment_data)/len(self.user_features)*100:.1f}%)")
            print(f"Avg interactions: {segment_data['total_interactions'].mean():.1f}")
            print(f"Avg time viewed: {segment_data['avg_time_viewed'].mean():.1f}s")
            print(f"Avg click rate: {segment_data['click_rate'].mean():.1%}")
            
            # Top content preferences
            content_cols = [col for col in segment_data.columns if col.startswith('pref_')]
            top_content = segment_data[content_cols].mean().sort_values(ascending=False).head(3)
            print(f"Top content preferences:")
            for content, pref in top_content.items():
                print(f"  - {content.replace('pref_', '')}: {pref:.1%}")
            
            # Business recommendations
            if segment == "Highly Engaged":
                print("Business Recommendations:")
                print("  - Provide premium, in-depth content")
                print("  - Offer exclusive features or early access")
                print("  - Focus on retention and loyalty programs")
            elif segment == "Active Users":
                print("Business Recommendations:")
                print("  - Diversify content offerings")
                print("  - Implement personalized notifications")
                print("  - A/B test different content formats")
            elif segment == "Browsers":
                print("Business Recommendations:")
                print("  - Optimize for mobile and quick scanning")
                print("  - Use more visual content and infographics")
                print("  - Implement progressive disclosure")
            else:  # Casual Users
                print("Business Recommendations:")
                print("  - Simplify onboarding process")
                print("  - Focus on clear value propositions")
                print("  - Use email/push notifications strategically")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive user engagement analysis...")
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Create user features
        self.create_user_features()
        
        # Perform segmentation
        self.perform_user_segmentation()
        
        # Analyze content preferences
        self.analyze_content_preferences()
        
        # Generate recommendations
        recommendations = self.generate_personalized_recommendations()
        
        # Create visualizations
        self.create_visualizations()
        
        # Print sample recommendations
        self.print_sample_recommendations(recommendations)
        
        # Generate segment insights
        self.generate_segment_insights()
        
        return recommendations

# Run the analysis
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = UserEngagementAnalyzer('user_engagement_final.csv')
    
    # Run complete analysis
    recommendations = analyzer.run_complete_analysis()
    
    # Save recommendations to CSV
    recommendations.to_csv('user_recommendations.csv', index=False)
    print(f"\nPersonalized recommendations saved to 'user_recommendations.csv'")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Key outputs:")
    print("1. User segmentation with 4 distinct segments")
    print("2. Content preferences analysis by segment")
    print("3. Personalized recommendations for each user")
    print("4. Visual dashboard of engagement patterns")
    print("5. Business recommendations for each segment")
