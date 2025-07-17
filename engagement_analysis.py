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
        """Load and explore the engagement data - """
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
        
        # Debug: Showing engagement score distribution
        print("\nDEBUG: Engagement score statistics:")
        print("Segment summary with calculated engagement scores:")
        for segment in segment_summary.index:
            stats = segment_summary.loc[segment]
            engagement_score = (stats['click_rate'] * 0.5 + 
                              min(stats['avg_time_viewed'] / 60, 1) * 0.25 + 
                              min(stats['total_interactions'] / 12, 1) * 0.25)
            segment_name = self.user_features[self.user_features['segment'] == segment]['segment_name'].iloc[0]
            user_count = len(self.user_features[self.user_features['segment'] == segment])
            print(f"  {segment_name}: {engagement_score:.3f} score, {user_count} users")
        
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
    
    def create_visualizations(self):
        """Create comprehensive visualizations - Enhanced Day 1 Analysis"""
        print("\n" + "=" * 60)
        print("CREATING ENHANCED VISUALIZATIONS")
        print("=" * 60)
        
        # Create multiple figure sets for comprehensive analysis
        
        # FIGURE 1: Main Dashboard (2x3 grid)
        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
        fig1.suptitle('User Engagement Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Segment distribution
        segment_counts = self.user_features['segment_name'].value_counts()
        axes1[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes1[0, 0].set_title('User Segment Distribution')
        
        # 2. Content type popularity
        content_popularity = self.df['content_type'].value_counts()
        axes1[0, 1].bar(content_popularity.index, content_popularity.values, color='skyblue')
        axes1[0, 1].set_title('Content Type Popularity')
        axes1[0, 1].tick_params(axis='x', rotation=45)
        axes1[0, 1].set_ylabel('Number of Interactions')
        
        # 3. Click rate by content type
        click_rates = self.df.groupby('content_type')['clicked'].mean().sort_values(ascending=False)
        axes1[0, 2].bar(click_rates.index, click_rates.values, color='lightcoral')
        axes1[0, 2].set_title('Click Rate by Content Type')
        axes1[0, 2].tick_params(axis='x', rotation=45)
        axes1[0, 2].set_ylabel('Click Rate')
        
        # 4. Average time viewed by segment
        avg_times = self.user_features.groupby('segment_name')['avg_time_viewed'].mean()
        axes1[1, 0].bar(avg_times.index, avg_times.values, color='lightgreen')
        axes1[1, 0].set_title('Average Time Viewed by Segment')
        axes1[1, 0].tick_params(axis='x', rotation=45)
        axes1[1, 0].set_ylabel('Seconds')
        
        # 5. Interactions vs Click Rate by segment
        for segment in self.user_features['segment_name'].unique():
            segment_data = self.user_features[self.user_features['segment_name'] == segment]
            axes1[1, 1].scatter(segment_data['total_interactions'], segment_data['click_rate'], 
                             label=segment, alpha=0.7, s=50)
        axes1[1, 1].set_xlabel('Total Interactions')
        axes1[1, 1].set_ylabel('Click Rate')
        axes1[1, 1].set_title('Interactions vs Click Rate by Segment')
        axes1[1, 1].legend()
        
        # 6. Content preference heatmap
        content_prefs = self.user_features.groupby('segment_name')[[col for col in self.user_features.columns if col.startswith('pref_')]].mean()
        content_prefs.columns = [col.replace('pref_', '') for col in content_prefs.columns]
        
        im = axes1[1, 2].imshow(content_prefs.values, cmap='YlOrRd', aspect='auto')
        axes1[1, 2].set_xticks(range(len(content_prefs.columns)))
        axes1[1, 2].set_xticklabels(content_prefs.columns, rotation=45)
        axes1[1, 2].set_yticks(range(len(content_prefs.index)))
        axes1[1, 2].set_yticklabels(content_prefs.index)
        axes1[1, 2].set_title('Content Preferences by Segment')
        plt.colorbar(im, ax=axes1[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        # FIGURE 2: Detailed Histograms and Distributions (2x2 grid)
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
        fig2.suptitle('Detailed Data Distributions', fontsize=16, fontweight='bold')
        
        # 1. Time viewed histogram
        axes2[0, 0].hist(self.df['time_viewed_in_sec'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes2[0, 0].set_title('Distribution of Time Viewed')
        axes2[0, 0].set_xlabel('Time Viewed (seconds)')
        axes2[0, 0].set_ylabel('Frequency')
        axes2[0, 0].axvline(self.df['time_viewed_in_sec'].mean(), color='red', linestyle='--', 
                           label=f'Mean: {self.df["time_viewed_in_sec"].mean():.1f}s')
        axes2[0, 0].legend()
        
        # 2. User interactions histogram
        user_interactions = self.df.groupby('user_id').size()
        axes2[0, 1].hist(user_interactions, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes2[0, 1].set_title('Distribution of User Interactions')
        axes2[0, 1].set_xlabel('Number of Interactions per User')
        axes2[0, 1].set_ylabel('Number of Users')
        axes2[0, 1].axvline(user_interactions.mean(), color='red', linestyle='--', 
                           label=f'Mean: {user_interactions.mean():.1f}')
        axes2[0, 1].legend()
        
        # 3. Click rate by engagement segment
        click_rates_by_segment = self.user_features.groupby('segment_name')['click_rate'].mean()
        axes2[1, 0].bar(click_rates_by_segment.index, click_rates_by_segment.values, 
                       color=['red', 'orange', 'green'])
        axes2[1, 0].set_title('Click Rate by Engagement Segment')
        axes2[1, 0].set_ylabel('Average Click Rate')
        axes2[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Time spent by content type (box plot)
        content_types = self.df['content_type'].unique()
        time_data = [self.df[self.df['content_type'] == ct]['time_viewed_in_sec'].values for ct in content_types]
        axes2[1, 1].boxplot(time_data, labels=content_types)
        axes2[1, 1].set_title('Time Viewed Distribution by Content Type')
        axes2[1, 1].set_ylabel('Time Viewed (seconds)')
        axes2[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # FIGURE 3: Engagement Analysis (2x2 grid)
        fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))
        fig3.suptitle('User Engagement Deep Dive', fontsize=16, fontweight='bold')
        
        # 1. Engagement score distribution
        engagement_scores = []
        for _, user_data in self.user_features.iterrows():
            score = (user_data['click_rate'] * 0.5 + 
                    min(user_data['avg_time_viewed'] / 60, 1) * 0.25 + 
                    min(user_data['total_interactions'] / 12, 1) * 0.25)
            engagement_scores.append(score)
        
        axes3[0, 0].hist(engagement_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes3[0, 0].set_title('Distribution of Engagement Scores')
        axes3[0, 0].set_xlabel('Engagement Score')
        axes3[0, 0].set_ylabel('Number of Users')
        
        # 2. Content type interactions over time (if we have dates)
        daily_interactions = self.df.groupby(['date', 'content_type']).size().unstack(fill_value=0)
        daily_interactions.plot(kind='line', ax=axes3[0, 1], marker='o', alpha=0.7)
        axes3[0, 1].set_title('Content Interactions Over Time')
        axes3[0, 1].set_xlabel('Date')
        axes3[0, 1].set_ylabel('Number of Interactions')
        axes3[0, 1].legend(title='Content Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes3[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. User activity heatmap by day
        df_copy = self.df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['day'] = df_copy['date'].dt.day_name()
        daily_activity = df_copy.groupby('day').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        axes3[1, 0].bar(daily_activity.index, daily_activity.values, color='orange')
        axes3[1, 0].set_title('Activity by Day of Week')
        axes3[1, 0].set_ylabel('Number of Interactions')
        axes3[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Correlation heatmap of user metrics
        user_metrics = self.user_features[['avg_time_viewed', 'click_rate', 'total_interactions', 'unique_content_viewed']].corr()
        im = axes3[1, 1].imshow(user_metrics.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes3[1, 1].set_xticks(range(len(user_metrics.columns)))
        axes3[1, 1].set_xticklabels(user_metrics.columns, rotation=45)
        axes3[1, 1].set_yticks(range(len(user_metrics.index)))
        axes3[1, 1].set_yticklabels(user_metrics.index)
        axes3[1, 1].set_title('User Metrics Correlation')
        
        # Add correlation values to heatmap
        for i in range(len(user_metrics.index)):
            for j in range(len(user_metrics.columns)):
                axes3[1, 1].text(j, i, f'{user_metrics.iloc[i, j]:.2f}', 
                               ha='center', va='center', color='white' if abs(user_metrics.iloc[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=axes3[1, 1], fraction=0.046, pad=0.04)
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
    
    def analyze_cluster_composition(self):
        """Analyze the composition of each K-means cluster by individual engagement levels"""
        print("\n" + "=" * 60)
        print("CLUSTER COMPOSITION ANALYSIS")
        print("=" * 60)
        
        # Calculate individual engagement scores for each user
        individual_scores = []
        for _, user_data in self.user_features.iterrows():
            score = (user_data['click_rate'] * 0.5 + 
                    min(user_data['avg_time_viewed'] / 60, 1) * 0.25 + 
                    min(user_data['total_interactions'] / 12, 1) * 0.25)
            individual_scores.append(score)
        
        self.user_features['individual_engagement_score'] = individual_scores
        
        # Define thresholds for individual engagement classification (using percentiles)
        low_threshold = np.percentile(individual_scores, 33.33)
        high_threshold = np.percentile(individual_scores, 66.67)
        
        print(f"Individual Engagement Score Thresholds:")
        print(f"  Low Engagement: < {low_threshold:.3f}")
        print(f"  Medium Engagement: {low_threshold:.3f} - {high_threshold:.3f}")
        print(f"  High Engagement: > {high_threshold:.3f}")
        
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
        
        # Create cross-tabulation of clusters vs individual engagement levels
        print(f"\nCLUSTER COMPOSITION BY INDIVIDUAL ENGAGEMENT LEVELS:")
        print("=" * 50)
        
        composition = pd.crosstab(
            self.user_features['segment'], 
            self.user_features['individual_engagement_level'], 
            margins=True
        )
        print(composition)
        
        # Show percentages within each cluster
        print(f"\nPERCENTAGE BREAKDOWN WITHIN EACH CLUSTER:")
        print("=" * 50)
        composition_pct = pd.crosstab(
            self.user_features['segment'], 
            self.user_features['individual_engagement_level'], 
            normalize='index'
        ) * 100
        print(composition_pct.round(1))
        
        # Detailed breakdown for each cluster
        print(f"\nDETAILED CLUSTER BREAKDOWN:")
        print("=" * 50)
        
        for cluster in sorted(self.user_features['segment'].unique()):
            cluster_data = self.user_features[self.user_features['segment'] == cluster]
            cluster_name = cluster_data['segment_name'].iloc[0]
            total_users = len(cluster_data)
            
            print(f"\nCluster {cluster} ({cluster_name}) - {total_users} users:")
            
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
    
    def visualize_clusters(self):
        """Visualize the K-means clusters using PCA"""
        print("\n" + "=" * 60)
        print("CLUSTER VISUALIZATION")
        print("=" * 60)
        
        # Prepare features for clustering (same as used in segmentation)
        features_for_clustering = self.user_features.select_dtypes(include=[np.number]).fillna(0)
        # Remove the segment column and any other non-feature columns
        feature_cols = [col for col in features_for_clustering.columns 
                       if col not in ['segment', 'individual_engagement_score']]
        features_clean = features_for_clustering[feature_cols]
        
        # Scale features (same as used in clustering)
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create the visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-means Clustering Visualization', fontsize=16, fontweight='bold')
        
        # 1. Clusters by segment number (0, 1, 2)
        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=self.user_features['segment'], 
                              cmap='viridis', alpha=0.7, s=50)
        ax1.set_title('Clusters by Segment Number')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # Add cluster centers
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        centers_2d = pca.transform(kmeans.cluster_centers_)
        ax1.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        ax1.legend()
        
        # 2. Clusters by engagement level names
        segment_colors = {'High Engagement': 'red', 'Medium Engagement': 'orange', 'Low Engagement': 'blue'}
        for segment_name, color in segment_colors.items():
            mask = self.user_features['segment_name'] == segment_name
            if mask.any():
                ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=color, label=segment_name, alpha=0.7, s=50)
        ax2.set_title('Clusters by Engagement Level Names')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax2.legend()
        
        # 3. Individual engagement scores (if available)
        if 'individual_engagement_score' in self.user_features.columns:
            scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=self.user_features['individual_engagement_score'], 
                                  cmap='RdYlBu_r', alpha=0.7, s=50)
            ax3.set_title('Individual Engagement Scores')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter3, ax=ax3, label='Engagement Score')
        else:
            ax3.text(0.5, 0.5, 'Individual engagement scores\nnot calculated yet', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Individual Engagement Scores (Not Available)')
        
        # 4. Feature importance (PCA components)
        feature_names = feature_cols
        pc1_importance = np.abs(pca.components_[0])
        pc2_importance = np.abs(pca.components_[1])
        
        # Get top 10 most important features for each PC
        top_pc1_idx = np.argsort(pc1_importance)[-10:]
        top_pc2_idx = np.argsort(pc2_importance)[-10:]
        
        y_pos = np.arange(len(top_pc1_idx))
        ax4.barh(y_pos, pc1_importance[top_pc1_idx], alpha=0.7, label='PC1')
        ax4.barh(y_pos + 0.4, pc2_importance[top_pc1_idx], alpha=0.7, label='PC2')
        ax4.set_yticks(y_pos + 0.2)
        ax4.set_yticklabels([feature_names[i][:15] + '...' if len(feature_names[i]) > 15 
                            else feature_names[i] for i in top_pc1_idx])
        ax4.set_xlabel('Absolute Component Weight')
        ax4.set_title('Top Features Contributing to PCs')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print PCA explanation
        print(f"PCA Explained Variance:")
        print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"  Total: {sum(pca.explained_variance_ratio_):.1%}")
        
        return features_2d, pca
    
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
        
        # Analyze cluster composition
        self.analyze_cluster_composition()

        # Visualize clusters
        self.visualize_clusters()
        
        return recommendations

# Run the analysis
if __name__ == "__main__":
    #Merge/join the data on user_id
    engagement = pd.read_csv('user_engagement_final.csv')
    attributes = pd.read_csv('user_attributes.csv')
    # Join on 'user_id'
    merged = pd.merge(engagement, attributes, on='user_id', how='inner')
    merged.to_csv('joined_output.csv', index=False)  # or 'l
    
    df = pd.read_csv('joined_output.csv')

    print(f"Original shape: {df.shape}")
    print(f"First few user_ids: {df['user_id'].head().tolist()}")

    # Sort by user_id
    df_sorted = df.sort_values('user_id')

    print(f"After sorting - First few user_ids: {df_sorted['user_id'].head().tolist()}")

    # Save the sorted version
    df_sorted.to_csv('joined_user_table.csv', index=False)
    # Initialize analyzer
    analyzer = UserEngagementAnalyzer('joined_user_table.csv')
    
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
    print("4. Visual dashboard of engagement patterns")
    print("5. Business recommendations for each engagement tier")
