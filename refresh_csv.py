import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def regenerate_recommendations():
    """Quick script to regenerate just the recommendations CSV"""
    
    # Load data
    df = pd.read_csv('user_engagement_final.csv')
    
    # Create user features (simplified version)
    user_stats = df.groupby('user_id').agg({
        'time_viewed_in_sec': ['mean', 'sum', 'count'],
        'clicked': ['mean', 'sum'],
        'content_id': 'nunique'
    }).round(2)
    
    user_stats.columns = ['avg_time_viewed', 'total_time_viewed', 'total_interactions',
                         'click_rate', 'total_clicks', 'unique_content_viewed']
    
    # Content type preferences
    content_preferences = df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
    content_preferences = content_preferences.div(content_preferences.sum(axis=1), axis=0)
    content_preferences.columns = [f'pref_{col}' for col in content_preferences.columns]
    
    # Average time by content type
    avg_time_by_content = df.groupby(['user_id', 'content_type'])['time_viewed_in_sec'].mean().unstack(fill_value=0)
    avg_time_by_content.columns = [f'avg_time_{col}' for col in avg_time_by_content.columns]
    
    # Click rates by content type
    click_rates_by_content = df.groupby(['user_id', 'content_type'])['clicked'].mean().unstack(fill_value=0)
    click_rates_by_content.columns = [f'click_rate_{col}' for col in click_rates_by_content.columns]
    
    # Combine features
    user_features = pd.concat([
        user_stats,
        content_preferences,
        avg_time_by_content,
        click_rates_by_content
    ], axis=1).fillna(0)
    
    # Perform clustering
    scaler = StandardScaler()
    features_for_clustering = user_features.select_dtypes(include=[np.number]).fillna(0)
    features_scaled = scaler.fit_transform(features_for_clustering)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    user_features['segment'] = kmeans.fit_predict(features_scaled)
    
    # Calculate engagement scores and rank segments
    segment_summary = user_features.groupby('segment').agg({
        'total_interactions': 'mean',
        'avg_time_viewed': 'mean',
        'click_rate': 'mean'
    }).round(2)
    
    segment_scores = {}
    for segment in segment_summary.index:
        stats = segment_summary.loc[segment]
        engagement_score = (stats['click_rate'] * 0.5 + 
                          min(stats['avg_time_viewed'] / 60, 1) * 0.25 + 
                          min(stats['total_interactions'] / 12, 1) * 0.25)
        segment_scores[segment] = engagement_score
    
    # Sort and assign names
    sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
    segment_names = {}
    
    for i, (segment, score) in enumerate(sorted_segments):
        if i == 0:
            segment_names[segment] = "High Engagement"
        elif i == 1:
            segment_names[segment] = "Medium Engagement"
        else:
            segment_names[segment] = "Low Engagement"
    
    user_features['segment_name'] = user_features['segment'].map(segment_names)
    
    print("Segment distribution:")
    print(user_features['segment_name'].value_counts())
    
    # Generate recommendations
    recommendations = []
    content_types = ['improve', 'insights', 'drivescore', 'protect', 'credit_cards', 'loans']
    
    for user_id, user_data in user_features.iterrows():
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
            pref_score = user_prefs[content_type] * 0.3
            time_score = min(user_avg_times[content_type] / 60, 1) * 0.4
            click_score = user_click_rates[content_type] * 0.3
            rec_scores[content_type] = pref_score + time_score + click_score
        
        # Sort by recommendation score
        sorted_content = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate strategy based on segment
        if segment == "High Engagement":
            strategy = "Provide premium, in-depth content with comprehensive details and advanced features"
        elif segment == "Medium Engagement":
            strategy = "Offer balanced content with clear value propositions and moderate detail"
        else:
            strategy = "Provide simple, bite-sized content with clear benefits and easy next steps"
        
        recommendations.append({
            'user_id': user_id,
            'segment': segment,
            'primary_recommendation': sorted_content[0][0],
            'secondary_recommendation': sorted_content[1][0],
            'strategy': strategy,
            'user_click_rate': user_data['click_rate'],
            'user_avg_time': user_data['avg_time_viewed'],
            'total_interactions': user_data['total_interactions']
        })
    
    # Save to CSV
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv('user_recommendations.csv', index=False)
    print(f"\nSaved {len(recommendations_df)} user recommendations to 'user_recommendations.csv'")
    
    # Show sample
    print("\nSample of updated recommendations:")
    print(recommendations_df[['user_id', 'segment', 'primary_recommendation']].head(10))
    
    return recommendations_df

if __name__ == "__main__":
    regenerate_recommendations() 