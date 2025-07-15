from engagement_analysis import UserEngagementAnalyzer

print("Starting test...")

try:
    # Initialize analyzer
    analyzer = UserEngagementAnalyzer('user_engagement_final.csv')
    print("✓ Analyzer initialized")
    
    # Load and explore data
    analyzer.load_and_explore_data()
    print("✓ Data loaded")
    
    # Create user features
    analyzer.create_user_features()
    print("✓ Features created")
    
    # Perform segmentation
    analyzer.perform_user_segmentation()
    print("✓ Segmentation complete")
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc() 