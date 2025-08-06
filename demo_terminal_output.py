#!/usr/bin/env python3
"""
Demo script to showcase the new terminal output functionality
Shows detailed user distribution summaries for segments and financial health categories
"""

from fixed_advanced_recommender import FixedAdvancedRecommender, SystemConfig, FinancialConfig, EngagementConfig

def demo_terminal_output():
    """Demonstrate the detailed terminal output"""
    print("🎯 DEMONSTRATING ENHANCED TERMINAL OUTPUT")
    print("=" * 60)
    print("This demo shows how the system now prints detailed user distribution")
    print("information directly to the terminal, including:")
    print("• User counts per financial health category")
    print("• User counts per enhanced segment")
    print("• Statistical summaries and key insights")
    print("=" * 60)
    
    # Demo 1: Standard configuration
    print("\n🔧 DEMO 1: Standard Configuration")
    print("-" * 40)
    standard_recommender = FixedAdvancedRecommender('joined_user_table.csv')
    print("Creating user features... (watch for the distribution summary)")
    data = standard_recommender.create_user_features()
    
    # Demo 2: Custom configuration with different thresholds
    print("\n🔧 DEMO 2: Custom Configuration (More Strict Thresholds)")
    print("-" * 40)
    custom_config = SystemConfig(
        financial=FinancialConfig(
            excellent_threshold=0.9,  # More strict
            good_threshold=0.75,
            fair_threshold=0.55
        ),
        engagement=EngagementConfig(
            high_engagement_threshold=0.7,  # Higher bar
            medium_engagement_threshold=0.4
        )
    )
    
    custom_recommender = FixedAdvancedRecommender('joined_user_table.csv', config=custom_config)
    print("Creating user features with stricter thresholds...")
    custom_data = custom_recommender.create_user_features()
    
    # Demo 3: Show how visualization methods include summaries
    print("\n🔧 DEMO 3: Visualization Methods Include Distribution Info")
    print("-" * 40)
    print("When creating visualizations, you'll see quick summaries...")
    
    # Just create one visualization to show the summary output
    print("\n📊 Creating financial dashboard (watch for quick summary):")
    standard_recommender.create_financial_visualizations()
    
    print("\n📊 Creating clustering visualization (watch for quick summary):")
    standard_recommender.visualize_financial_clustering()
    
    print("\n📊 Creating correlation analysis (watch for quick summary):")
    standard_recommender.create_engagement_financial_correlation()
    
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("✅ As you can see, the system now provides rich terminal output")
    print("✅ User distributions are clearly shown with counts and percentages")
    print("✅ Key insights highlight important patterns in the data")
    print("✅ Quick summaries appear with each visualization method")
    print("\nThis makes it much easier to understand your user base")
    print("without having to look at the charts!")

if __name__ == "__main__":
    demo_terminal_output() 