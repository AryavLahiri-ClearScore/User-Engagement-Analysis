import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class UserEngagementAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.user_features = None
        self.segments = None
        self.scaler = StandardScaler()
        
    def load_and_explore_data(self):
        """Load and explore the engagement data"""
        print("=" * 60)
        print("DAY 1: EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Number of unique users: {self.df['user_id'].nunique()}")
        print(f"Number of unique content pieces: {self.df['content_id'].nunique()}")
        
        # Most common content types
        print("\n1. MOST COMMON CONTENT TYPES:")
        content_counts = self.df['content_type'].value_counts()
        for content_type, count in content_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {content_type}: {count} interactions ({percentage:.1f}%)")
        
        #Time spent analysis
        print("\n2. TIME SPENT ON PAGES:")
        time_stats = self.df.groupby('content_type')['time_viewed_in_sec'].agg(['mean', 'median', 'std']).round(1)
        print("   Average time by content type:")
        for content_type in time_stats.index:
            print(f"   {content_type}: {time_stats.loc[content_type, 'mean']}s avg, "
                  f"{time_stats.loc[content_type, 'median']}s median")
        
        #Click rate analysis
        print("\n3. CLICK RATES BY CONTENT TYPE:")
        click_rates = self.df.groupby('content_type')['clicked'].mean().sort_values(ascending=False)
        for content_type, rate in click_rates.items():
            print(f"   {content_type}: {rate:.1%} click rate")
        
        print(f"\n   Overall click rate: {self.df['clicked'].mean():.2%}")
        print(f"   Total clicks: {self.df['clicked'].sum():,} out of {len(self.df):,} interactions")
        
        #User engagement frequency
        print("\n4. USER ENGAGEMENT FREQUENCY:")
        user_interactions = self.df.groupby('user_id').size()
        print(f"   Average interactions per user: {user_interactions.mean():.1f}")
        print(f"   Median interactions per user: {user_interactions.median():.1f}")
        print(f"   Most active user: {user_interactions.max()} interactions")
        print(f"   Least active user: {user_interactions.min()} interactions")
        
        # Additional insights
        print("\n5. ADDITIONAL INSIGHTS:")
        print(f"   Average time viewed overall: {self.df['time_viewed_in_sec'].mean():.1f} seconds")
        print(f"   Median time viewed: {self.df['time_viewed_in_sec'].median():.1f} seconds")
        print(f"   Longest session: {self.df['time_viewed_in_sec'].max():.1f} seconds")
        
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
    
    def perform_user_segmentation(self, n_clusters=3):
        """DAY 2: Perform user segmentation using K-means clustering"""
        print("\n" + "=" * 60)
        print("DAY 2: USER SEGMENTATION")
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
        
        # Debug: Showing actual user counts per segment
        print("\nDEBUG: Actual user counts per segment:")
        segment_counts = self.user_features['segment_name'].value_counts()
        print(segment_counts)
        
        return self.user_features
    
    def name_segments(self, segment_summary):
        """Name segments by directly ranking them by engagement score"""
        segment_names = {}
        
        print("\nDEBUG: Segment characteristics before naming:")
        print(segment_summary)
        
        # Calculate engagement scores for all segments
        segment_scores = {}
        for segment in segment_summary.index:
            stats = segment_summary.loc[segment]
            engagement_score = (stats['click_rate'] * 0.5 + 
                              min(stats['avg_time_viewed'] / 60, 1) * 0.25 + 
                              min(stats['total_interactions'] / 12, 1) * 0.25)
            segment_scores[segment] = engagement_score
        
        # Sort segments by engagement score (highest to lowest)
        sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nDEBUG: Segments ranked by engagement score:")
        for i, (segment, score) in enumerate(sorted_segments):
            print(f"  Rank {i+1}: Segment {segment} - Score: {score:.3f}")
        
        # Assign names based on ranking (guaranteed distribution)
        for i, (segment, score) in enumerate(sorted_segments):
            stats = segment_summary.loc[segment]
            
            print(f"\nSegment {segment}:")
            print(f"  Click rate: {stats['click_rate']:.3f}")
            print(f"  Avg time viewed: {stats['avg_time_viewed']:.1f}s")
            print(f"  Total interactions: {stats['total_interactions']:.1f}")
            print(f"  Engagement Score: {score:.3f}")
            
            # Assign based on rank (ensures all 3 tiers are used)
            if i == 0:  # Highest scoring cluster
                segment_names[segment] = "High Engagement"
                print(f"  -> Assigned: High Engagement (Rank 1)")
            elif i == 1:  # Middle scoring cluster
                segment_names[segment] = "Medium Engagement"
                print(f"  -> Assigned: Medium Engagement (Rank 2)")
            else:  # Lowest scoring cluster
                segment_names[segment] = "Low Engagement"
                print(f"  -> Assigned: Low Engagement (Rank 3)")
        
        print("Segment names: ", segment_names)
        return segment_names
    
    def generate_personalized_recommendations(self):
        """DAY 3: Generate personalized content recommendations for each user"""
        print("\n" + "=" * 60)
        print("DAY 3: PERSONALIZED CONTENT RECOMMENDATIONS")
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
            if segment == "High Engagement":
                primary_rec = sorted_content[0][0]
                secondary_rec = sorted_content[1][0]
                strategy = "Provide premium, in-depth content with comprehensive details and advanced features"
            elif segment == "Medium Engagement":
                primary_rec = sorted_content[0][0]
                secondary_rec = sorted_content[1][0]
                strategy = "Offer balanced content with clear value propositions and moderate detail"
            else:  # Low Engagement
                primary_rec = sorted_content[0][0]
                secondary_rec = sorted_content[1][0]
                strategy = "Provide simple, bite-sized content with clear benefits and easy next steps"
            
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
            if segment == "High Engagement":
                print("Business Recommendations:")
                print("  - Provide premium, in-depth content and whitepapers")
                print("  - Offer exclusive features, early access, and VIP treatment")
                print("  - Focus on retention and loyalty programs")
                print("  - Create detailed guides, tutorials, and educational content")
                print("  - Implement advanced personalization features")
            elif segment == "Medium Engagement":
                print("Business Recommendations:")
                print("  - Offer balanced content with clear value propositions")
                print("  - Implement targeted notifications and reminders")
                print("  - A/B test different content formats and lengths")
                print("  - Provide moderate detail with easy-to-scan formatting")
                print("  - Focus on conversion optimization")
            else:  # Low Engagement
                print("Business Recommendations:")
                print("  - Simplify onboarding and user experience")
                print("  - Focus on clear, immediate value propositions")
                print("  - Use prominent CTAs and minimal friction")
                print("  - Create bite-sized, easy-to-consume content")
                print("  - Implement re-engagement campaigns and incentives")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline WITHOUT visualizations"""
        print("Starting comprehensive user engagement analysis...")
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Create user features
        self.create_user_features()
        
        # Perform segmentation
        self.perform_user_segmentation()
        
        # Generate recommendations
        recommendations = self.generate_personalized_recommendations()
        
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
    print("1. User segmentation with 3 engagement tiers:")
    print("   - High Engagement: Users with strong click rates, time spent, and interactions")
    print("   - Medium Engagement: Users with moderate engagement across metrics")
    print("   - Low Engagement: Users with minimal engagement requiring activation")
    print("2. Content preferences analysis by engagement level")
    print("3. Personalized recommendations for each user")
    print("4. Visual dashboard of engagement patterns (SKIPPED - no visualizations)")
    print("5. Business recommendations for each engagement tier") 