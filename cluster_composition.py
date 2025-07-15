from engagement_analysis import UserEngagementAnalyzer

# Initialize analyzer
analyzer = UserEngagementAnalyzer('user_engagement_final.csv')

# Load data and create features
analyzer.load_and_explore_data()
analyzer.create_user_features() 
analyzer.perform_user_segmentation()

# Run the cluster composition analysis
composition, composition_pct = analyzer.analyze_cluster_composition()

# Visualize the clusters
features_2d, pca = analyzer.visualize_clusters() 