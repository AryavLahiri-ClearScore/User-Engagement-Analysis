import pandas as pd

# Read the current recommendations
current_recs = pd.read_csv('user_recommendations.csv')

print("Current Cluster-to-Engagement mapping:")
print("=" * 50)

# If we can get cluster numbers, show the mapping
# (This assumes the CSV has both cluster numbers and segment names)

# Show some sample users to see the pattern
print("\nSample of current classifications:")
print(current_recs[['user_id', 'segment']].head(10))

print(f"\nCurrent distribution:")
print(current_recs['segment'].value_counts())

print(f"\nTotal users: {len(current_recs)}")

# Check if there's any pattern in user IDs vs segments
print(f"\nFirst 5 High Engagement users: {current_recs[current_recs['segment'] == 'High Engagement']['user_id'].head().tolist()}")
print(f"First 5 Medium Engagement users: {current_recs[current_recs['segment'] == 'Medium Engagement']['user_id'].head().tolist()}")
print(f"First 5 Low Engagement users: {current_recs[current_recs['segment'] == 'Low Engagement']['user_id'].head().tolist()}") 