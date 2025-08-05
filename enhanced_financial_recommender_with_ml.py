# Enhanced Financial Recommender with ML Weight Optimization
# This shows how to integrate the ML optimizer into your main system

from enhanced_financial_recommender import FinanciallyAwareRecommender
from ml_weight_optimizer import MLWeightOptimizer
import pandas as pd
import numpy as np

class MLEnhancedFinancialRecommender(FinanciallyAwareRecommender):
    """
    Enhanced recommender that uses ML to optimize engagement weights
    """
    
    def __init__(self, csv_file, use_ml_weights=True):
        super().__init__(csv_file)
        self.use_ml_weights = use_ml_weights
        self.ml_weights = None
        self.weight_optimizer = None
        
    def optimize_engagement_weights(self, method='supervised', target='financial_health_score'):
        """
        Use ML to find optimal engagement score weights
        """
        print("\n🤖 OPTIMIZING ENGAGEMENT WEIGHTS WITH ML")
        print("=" * 60)
        
        if self.user_features is None:
            print("Creating user features first...")
            self.create_enhanced_user_features()
            
        # Initialize ML optimizer
        self.weight_optimizer = MLWeightOptimizer(self.user_features)
        
        # Choose optimization method
        if method == 'supervised':
            self.ml_weights = self.weight_optimizer.method_1_supervised_learning(target)
        elif method == 'pca':
            self.ml_weights = self.weight_optimizer.method_2_pca_weights()
        elif method == 'genetic':
            self.ml_weights = self.weight_optimizer.method_3_genetic_algorithm(target)
        elif method == 'multi_objective':
            self.ml_weights = self.weight_optimizer.method_4_multi_objective_optimization()
        elif method == 'compare_all':
            all_weights = self.weight_optimizer.compare_all_methods()
            # Use the supervised learning result as default
            self.ml_weights = all_weights[1]  # Index 1 = supervised learning
        else:
            raise ValueError(f"Unknown method: {method}")
            
        print(f"\n✅ ML optimization complete! Using {method} method.")
        return self.ml_weights
    
    def calculate_ml_engagement_score(self, user_data):
        """
        Calculate engagement score using ML-optimized weights
        """
        if self.ml_weights is None:
            # Fall back to manual weights if ML not run
            return (
                user_data['click_rate'] * 0.4 + 
                min(user_data['avg_time_viewed'] / 60, 1) * 0.3 + 
                min(user_data['total_interactions'] / 12, 1) * 0.3
            )
        else:
            # Use ML-optimized weights
            return (
                user_data['click_rate'] * self.ml_weights['click_rate'] +
                min(user_data['avg_time_viewed'] / 60, 1) * self.ml_weights['avg_time_viewed'] +
                min(user_data['total_interactions'] / 12, 1) * self.ml_weights['total_interactions']
            )
    
    def perform_enhanced_segmentation_with_ml(self):
        """
        Enhanced segmentation using ML-optimized engagement scores
        """
        print("Performing enhanced segmentation with ML-optimized weights...")
        
        # Select features for clustering (engagement + financial)
        engagement_features = ['avg_time_viewed', 'total_interactions', 'click_rate', 'unique_content_viewed']
        financial_features = ['financial_health_score', 'credit_score', 'dti_ratio', 'income']
        content_features = [col for col in self.user_features.columns if col.startswith('pref_')]
        
        clustering_features = engagement_features + financial_features + content_features
        features_for_clustering = self.user_features[clustering_features].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_for_clustering)
        
        # Perform K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        self.user_features['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Create ML-optimized engagement scores
        engagement_scores = []
        for _, user_data in self.user_features.iterrows():
            if self.use_ml_weights and self.ml_weights is not None:
                engagement_score = self.calculate_ml_engagement_score(user_data)
            else:
                # Original manual formula
                engagement_score = (
                    user_data['click_rate'] * 0.4 + 
                    min(user_data['avg_time_viewed'] / 60, 1) * 0.3 + 
                    min(user_data['total_interactions'] / 12, 1) * 0.3
                )
            engagement_scores.append(engagement_score)
        
        self.user_features['engagement_score'] = engagement_scores
        
        # Rest of segmentation logic (same as original)
        def assign_segment_name(row):
            engagement = row['engagement_score']
            financial_cat = row['financial_category']
            dti_ratio = row['dti_ratio']
            missed_payments = row['missed_payments']
            
            if missed_payments >= 2:
                return "Payment_Recovery_Priority"
            elif dti_ratio >= 0.5:
                return "Debt_Management_Priority"
            elif engagement > 0.5:
                if financial_cat == "Excellent":
                    return "Premium_Engaged"
                elif financial_cat in ["Good", "Fair"]:
                    return "Growth_Focused"
                else:
                    return "Recovery_Engaged"
            elif engagement > 0.25:
                if financial_cat == "Excellent":
                    return "Premium_Moderate"
                elif financial_cat in ["Good", "Fair"]:
                    return "Mainstream"
                else:
                    return "Recovery_Moderate"
            else:
                if financial_cat == "Poor":
                    return "Financial_Priority"
                else:
                    return "Activation_Needed"
        
        self.user_features['enhanced_segment'] = self.user_features.apply(assign_segment_name, axis=1)
        
        print("Enhanced segment distribution with ML-optimized engagement:")
        print(self.user_features['enhanced_segment'].value_counts())
        
        # Compare with manual weights
        if self.ml_weights is not None:
            print(f"\nML-Optimized Weights Used:")
            for feature, weight in self.ml_weights.items():
                print(f"  {feature}: {weight:.3f}")
        
        return self.user_features
    
    def run_ml_enhanced_analysis(self, optimization_method='compare_all'):
        """
        Run complete analysis with ML weight optimization
        """
        print("🚀 STARTING ML-ENHANCED ANALYSIS")
        print("=" * 70)
        
        # Step 1: Create base features
        self.create_enhanced_user_features()
        
        # Step 2: Optimize engagement weights using ML
        if self.use_ml_weights:
            self.optimize_engagement_weights(method=optimization_method)
        
        # Step 3: Perform segmentation with optimized weights
        if self.use_ml_weights:
            self.perform_enhanced_segmentation_with_ml()
        else:
            self.perform_enhanced_segmentation()
        
        # Step 4: Rest of analysis (same as original)
        recommendations = self.generate_financial_content_recommendations()
        
        # Step 5: Visualizations
        self.create_financial_visualizations()
        self.visualize_financial_clustering()
        self.create_engagement_financial_correlation()
        
        # Step 6: Print results
        self.print_enhanced_recommendations(recommendations)
        self.analyze_segments_by_financial_health()
        
        return recommendations


# Example usage
if __name__ == "__main__":
    print("🔬 ML-ENHANCED FINANCIAL RECOMMENDER")
    print("=" * 60)
    
    try:
        # Option 1: Run with ML optimization
        print("\n📊 RUNNING WITH ML WEIGHT OPTIMIZATION")
        ml_recommender = MLEnhancedFinancialRecommender(
            'joined_user_table.csv', 
            use_ml_weights=True
        )
        
        # This will compare all ML methods and use the best one
        ml_recommendations = ml_recommender.run_ml_enhanced_analysis(
            optimization_method='compare_all'
        )
        
        # Save results
        ml_recommendations.to_csv('ml_enhanced_recommendations.csv', index=False)
        print("\n✅ ML-enhanced recommendations saved to 'ml_enhanced_recommendations.csv'")
        
        # Option 2: Compare with manual weights
        print("\n📊 COMPARING WITH MANUAL WEIGHTS")
        manual_recommender = MLEnhancedFinancialRecommender(
            'joined_user_table.csv', 
            use_ml_weights=False
        )
        
        manual_recommendations = manual_recommender.run_enhanced_analysis()
        
        # Analysis comparison
        print("\n🔍 COMPARING SEGMENTATION RESULTS:")
        print("=" * 50)
        
        ml_segments = ml_recommendations['enhanced_segment'].value_counts()
        manual_segments = manual_recommendations['enhanced_segment'].value_counts()
        
        print("ML-Optimized Segments:")
        print(ml_segments)
        print("\nManual Weight Segments:")
        print(manual_segments)
        
        print("\n💡 Consider A/B testing these different approaches in production!")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Run the main enhanced_financial_recommender.py first to generate the data") 