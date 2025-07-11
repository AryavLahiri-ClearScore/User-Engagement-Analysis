import pandas as pd

def show_user_engagement_counts():
    """Show how many times each user engages"""
    
    # Load data
    df = pd.read_csv('user_engagement_final.csv')
    
    # Calculate interactions per user
    user_interactions = df.groupby('user_id').size().reset_index()
    user_interactions.columns = ['user_id', 'total_interactions']
    
    # Sort by most active users first
    user_interactions = user_interactions.sort_values('total_interactions', ascending=False)
    
    print("USER ENGAGEMENT COUNTS")
    print("=" * 40)
    print(f"Total users: {len(user_interactions)}")
    print(f"Average interactions per user: {user_interactions['total_interactions'].mean():.1f}")
    print(f"Most active user: {user_interactions['total_interactions'].max()} interactions")
    print(f"Least active user: {user_interactions['total_interactions'].min()} interactions")
    
    print(f"\nTOP 10 MOST ACTIVE USERS:")
    print(user_interactions.head(10).to_string(index=False))
    
    print(f"\nTOP 10 LEAST ACTIVE USERS:")
    print(user_interactions.tail(10).to_string(index=False))
    
    print(f"\nFULL LIST OF ALL USERS:")
    print(user_interactions.to_string(index=False))
    
    # Save to CSV if needed
    user_interactions.to_csv('user_engagement_counts.csv', index=False)
    print(f"\nSaved to 'user_engagement_counts.csv'")
    
    return user_interactions

if __name__ == "__main__":
    show_user_engagement_counts() 