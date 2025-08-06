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
        print(f"\n🤖 OPTIMIZING ENGAGEMENT WEIGHTS WITH ML ({method.upper()})")
        print("=" * 60)
        
        if self.user_features is None:
            print("Creating user features first...")
            self.create_enhanced_user_features()
            
        # Initialize ML optimizer
        self.weight_optimizer = MLWeightOptimizer(self.user_features)
        
        # Choose optimization method
        if method == 'supervised':
            print("🔬 Running Supervised Learning optimization...")
            self.ml_weights = self.weight_optimizer.method_1_supervised_learning(target)
        elif method == 'pca':
            print("🔬 Running PCA weights optimization...")
            self.ml_weights = self.weight_optimizer.method_2_pca_weights()
        elif method == 'genetic':
            print("🔬 Running Genetic Algorithm optimization...")
            self.ml_weights = self.weight_optimizer.method_3_genetic_algorithm(target)
        elif method == 'multi_objective':
            print("🔬 Running Multi-objective optimization...")
            self.ml_weights = self.weight_optimizer.method_4_multi_objective_optimization()
        elif method == 'compare_all':
            print("🔬 Running ALL optimization methods...")
            all_weights = self.weight_optimizer.compare_all_methods()
            
            # Debug: Show weights from all methods
            print("\n🔍 DEBUG: WEIGHTS FROM ALL METHODS:")
            method_names = ['supervised', 'pca', 'genetic', 'multi_objective']
            for i, weights in enumerate(all_weights):
                if i < len(method_names):
                    print(f"   {method_names[i].upper():15}: {weights}")
            
            # Use the supervised learning result as default
            self.ml_weights = all_weights[1]  # Index 1 = supervised learning
            print(f"\n📌 Selected SUPERVISED method weights: {self.ml_weights}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Debug: Show the final optimized weights
        if self.ml_weights:
            print(f"\n🎯 FINAL OPTIMIZED WEIGHTS ({method.upper()}):")
            print("-" * 50)
            total_weight = sum(self.ml_weights.values())
            for feature, weight in self.ml_weights.items():
                percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
                print(f"   {feature:20}: {weight:8.4f} ({percentage:5.1f}%)")
            print(f"   {'TOTAL':20}: {total_weight:8.4f} ({100.0:5.1f}%)")
            
            # Compare with manual weights
            manual_weights = {'click_rate': 0.4, 'avg_time_viewed': 0.3, 'total_interactions': 0.3}
            print(f"\n📊 COMPARISON WITH MANUAL WEIGHTS:")
            print("-" * 50)
            print(f"{'FEATURE':20} | {'MANUAL':>8} | {'ML':>8} | {'DIFF':>8}")
            print("-" * 50)
            for feature in manual_weights.keys():
                manual_val = manual_weights[feature]
                ml_val = self.ml_weights.get(feature, 0)
                diff = ml_val - manual_val
                print(f"{feature:20} | {manual_val:8.3f} | {ml_val:8.3f} | {diff:+8.3f}")
        else:
            print("❌ No weights were generated!")
            
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
    
    def run_single_method_analysis(self, method_name):
        """Run analysis with a single ML method"""
        print(f"\n🔬 ANALYZING WITH {method_name.upper()} METHOD")
        print("=" * 60)
        
        # Optimize weights with this specific method
        if method_name == 'manual':
            print("📊 Using manual weights (no ML optimization)")
            self.ml_weights = None
            
            # Show manual weights for comparison
            manual_weights = {'click_rate': 0.4, 'avg_time_viewed': 0.3, 'total_interactions': 0.3}
            print(f"\n📈 MANUAL WEIGHTS FOR {method_name.upper()}:")
            print("-" * 50)
            for feature, weight in manual_weights.items():
                percentage = (weight / 1.0) * 100
                print(f"   {feature:20}: {weight:8.4f} ({percentage:5.1f}%)")
        else:
            print(f"🤖 Optimizing engagement weights using {method_name}")
            self.optimize_engagement_weights(method=method_name)
            
            # The weights are already displayed in optimize_engagement_weights method
            # But let's add a summary here too
            if self.ml_weights:
                print(f"\n🔍 SUMMARY - {method_name.upper()} WEIGHTS:")
                print("-" * 40)
                for feature, weight in self.ml_weights.items():
                    print(f"   {feature:15}: {weight:7.4f}")
            else:
                print(f"❌ No weights generated for {method_name}!")
        
        # Perform segmentation with these weights
        if self.use_ml_weights and self.ml_weights:
            self.perform_enhanced_segmentation_with_ml()
        else:
            self.perform_enhanced_segmentation()
        
        # Generate recommendations
        recommendations = self.generate_financial_content_recommendations()
        
        # Show segment distribution in terminal
        print(f"\n📊 ENHANCED SEGMENTS ({method_name.upper()}):")
        segments = self.user_features['enhanced_segment'].value_counts()
        for segment, count in segments.items():
            percentage = (count / len(self.user_features)) * 100
            print(f"   {segment:25}: {count:3} users ({percentage:5.1f}%)")
        
        print(f"\n💰 FINANCIAL CATEGORIES ({method_name.upper()}):")
        financial_cats = self.user_features['financial_category'].value_counts()
        for category, count in financial_cats.items():
            percentage = (count / len(self.user_features)) * 100
            print(f"   {category:12}: {count:3} users ({percentage:5.1f}%)")
        
        # Create visualizations
        print(f"\n🎨 CREATING VISUALIZATIONS FOR {method_name.upper()}")
        print("-" * 50)
        
        try:
            self.create_financial_visualizations()
            print("   ✅ Financial Dashboard")
        except Exception as e:
            print(f"   ❌ Financial Dashboard failed: {e}")
        
        try:
            self.visualize_financial_clustering()
            print("   ✅ Clustering Visualization")
        except Exception as e:
            print(f"   ❌ Clustering failed: {e}")
        
        try:
            self.create_engagement_financial_correlation()
            print("   ✅ Correlation Analysis")
        except Exception as e:
            print(f"   ❌ Correlation failed: {e}")
        
        # Save method-specific recommendations
        filename = f"{method_name}_recommendations.csv"
        recommendations.to_csv(filename, index=False)
        print(f"💾 Saved: {filename}")
        
        return recommendations, segments, financial_cats
    
    def run_all_methods_analysis(self):
        """Run analysis with ALL 4 ML methods plus manual weights"""
        print("🚀 COMPREHENSIVE ALL-METHODS ANALYSIS")
        print("=" * 70)
        
        # Step 1: Create base features (once)
        print("\n1️⃣ CREATING BASE USER FEATURES")
        print("-" * 50)
        self.create_enhanced_user_features()
        print(f"✅ Created features for {len(self.user_features)} users")
        
        # Define all methods to test
        methods = ['manual', 'supervised', 'pca', 'genetic', 'multi_objective']
        
        all_results = {}
        
        # Run each method
        for method_name in methods:
            recommendations, segments, financial_cats = self.run_single_method_analysis(method_name)
            all_results[method_name] = {
                'recommendations': recommendations,
                'segments': segments,
                'financial_cats': financial_cats,
                'weights': self.ml_weights.copy() if self.ml_weights else None
            }
        
        # Final comparison summary
        print("\n" + "🏆" * 70)
        print("FINAL COMPARISON OF ALL METHODS")
        print("🏆" * 70)
        
        print("\n📊 ENHANCED SEGMENTS COMPARISON:")
        print("-" * 80)
        print(f"{'METHOD':15} | {'TOP SEGMENT':25} | {'COUNT':>5} | {'%':>6}")
        print("-" * 80)
        
        for method_name in methods:
            segments = all_results[method_name]['segments']
            if len(segments) > 0:
                top_segment = segments.index[0]
                count = segments.iloc[0]
                percentage = (count / len(self.user_features)) * 100
                print(f"{method_name.upper():15} | {top_segment:25} | {count:5} | {percentage:5.1f}%")
            else:
                print(f"{method_name.upper():15} | {'No segments found':25} | {'0':5} | {'0.0':>5}%")
        
        print(f"\n📋 DETAILED SEGMENTS FOR ALL METHODS:")
        print("-" * 80)
        
        for method_name in methods:
            segments = all_results[method_name]['segments']
            weights = all_results[method_name]['weights']
            
            print(f"\n🔬 {method_name.upper()} METHOD:")
            if weights:
                print(f"   Weights: click_rate={weights.get('click_rate', 0):.3f}, "
                      f"avg_time={weights.get('avg_time_viewed', 0):.3f}, "
                      f"interactions={weights.get('total_interactions', 0):.3f}")
            else:
                print(f"   Weights: Manual (click_rate=0.400, avg_time=0.300, interactions=0.300)")
            
            print(f"   Enhanced Segments:")
            for segment, count in segments.items():
                percentage = (count / len(self.user_features)) * 100
                print(f"     • {segment:25}: {count:3} users ({percentage:5.1f}%)")
        
        print(f"\n💰 FINANCIAL CATEGORIES COMPARISON:")
        print("-" * 80)
        
        for method_name in methods:
            financial_cats = all_results[method_name]['financial_cats']
            print(f"\n{method_name.upper():15} | Financial Categories:")
            for category, count in financial_cats.items():
                percentage = (count / len(self.user_features)) * 100
                print(f"{'':15} | {category:12}: {count:3} users ({percentage:5.1f}%)")
        
        print(f"\n✅ FILES GENERATED:")
        for method_name in methods:
            print(f"   - {method_name}_recommendations.csv")
        
        return all_results
    
    def run_ml_enhanced_analysis(self, optimization_method='all_methods'):
        """
        Run complete analysis - supports single method or all methods
        """
        if optimization_method == 'all_methods':
            return self.run_all_methods_analysis()
        else:
            # Single method analysis - for backwards compatibility
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
        
        # Run ALL 4 ML methods + manual for comprehensive comparison
        all_results = ml_recommender.run_ml_enhanced_analysis(
            optimization_method='all_methods'
        )
        
        print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("✅ All 4 ML methods + manual weights analyzed")
        print("✅ Enhanced segments shown for each method")
        print("✅ All visualizations created") 
        print("✅ All recommendation files saved")
        print("\n💡 Compare the segment distributions above to see how different")
        print("   ML optimization methods affect user segmentation!")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Run the main enhanced_financial_recommender.py first to generate the data") 