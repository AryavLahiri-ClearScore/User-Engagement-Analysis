import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FinanciallyAwareRecommender:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.user_features = None
        self.scaler = StandardScaler()
        
    def create_financial_health_score(self):
        """Create a comprehensive financial health score"""
        print("Creating financial health scores...")
        
        # Get unique users with their financial attributes
        user_financial = self.df.groupby('user_id').first()[
            ['total_debt', 'credit_score', 'missed_payments', 
             'has_mortgage', 'has_car', 'has_ccj', 'dti_ratio', 'income']
        ]
        
        financial_scores = []
        
        for _, user in user_financial.iterrows():
            # Credit Score Component (0-1, higher is better)
            credit_component = min(user['credit_score'] / 1000, 1.0)
            
            # Debt-to-Income Ratio Component (0-1, lower is better)
            dti_component = max(0, 1 - user['dti_ratio'])
            
            # Missed Payments Component (0-1, fewer is better)
            missed_payments_component = max(0, 1 - (user['missed_payments'] / 10))
            
            # Income Component (normalized, higher is better)
            income_component = min(user['income'] / 100000, 1.0)
            
            # CCJ Component (0-1, no CCJ is better)
            ccj_component = 0.0 if user['has_ccj'] else 1.0
            
            # Asset Component (having mortgage/car indicates stability)
            asset_component = (user['has_mortgage'] * 0.6 + user['has_car'] * 0.4)
            
            # Calculate composite financial health score (0-1)
            financial_health = (
                credit_component * 0.30 +
                dti_component * 0.25 +
                missed_payments_component * 0.15 +
                income_component * 0.15 +
                ccj_component * 0.10 +
                asset_component * 0.05
            )
            
            financial_scores.append(financial_health)
        
        user_financial['financial_health_score'] = financial_scores
        
        # Categorize financial health
        low_threshold = np.percentile(financial_scores, 33)
        high_threshold = np.percentile(financial_scores, 67)
        
        def categorize_financial_health(score):
            if score < low_threshold:
                return "At_Risk"
            elif score < high_threshold:
                return "Stable"
            else:
                return "Excellent"
                
        user_financial['financial_category'] = [
            categorize_financial_health(score) for score in financial_scores
        ]
        
        print(f"Financial health distribution:")
        print(user_financial['financial_category'].value_counts())
        
        return user_financial
    
    def create_enhanced_user_features(self):
        """Create user features including both engagement and financial data"""
        print("Creating enhanced user features...")
        
        # Basic engagement metrics
        user_stats = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).round(2)
        
        user_stats.columns = ['avg_time_viewed', 'total_time_viewed', 'total_interactions',
                             'click_rate', 'total_clicks', 'unique_content_viewed']
        
        # Content type preferences
        content_preferences = self.df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
        content_preferences = content_preferences.div(content_preferences.sum(axis=1), axis=0)
        content_preferences.columns = [f'pref_{col}' for col in content_preferences.columns]
        
        # Financial health scores
        financial_data = self.create_financial_health_score()
        
        # Combine all features
        self.user_features = pd.concat([
            user_stats,
            content_preferences,
            financial_data
        ], axis=1).fillna(0)
        
        print(f"Created {len(self.user_features.columns)} features for {len(self.user_features)} users")
        return self.user_features
    
    def perform_enhanced_segmentation(self):
        """Perform segmentation including financial health"""
        print("Performing enhanced segmentation...")
        
        # Select features for clustering (engagement + financial)
        engagement_features = ['avg_time_viewed', 'total_interactions', 'click_rate', 'unique_content_viewed']
        financial_features = ['financial_health_score', 'credit_score', 'dti_ratio', 'income']
        content_features = [col for col in self.user_features.columns if col.startswith('pref_')]
        
        clustering_features = engagement_features + financial_features + content_features
        features_for_clustering = self.user_features[clustering_features].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_for_clustering)
        
        # Perform K-means clustering with more clusters for nuanced segmentation
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        self.user_features['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Create composite engagement-financial score
        engagement_scores = []
        for _, user_data in self.user_features.iterrows():
            engagement_score = (
                user_data['click_rate'] * 0.4 + 
                min(user_data['avg_time_viewed'] / 60, 1) * 0.3 + 
                min(user_data['total_interactions'] / 12, 1) * 0.3
            )
            engagement_scores.append(engagement_score)
        
        self.user_features['engagement_score'] = engagement_scores
        
        # Assign meaningful segment names based on engagement and financial health
        def assign_segment_name(row):
            engagement = row['engagement_score']
            financial_cat = row['financial_category']
            
            if engagement > 0.5:
                if financial_cat == "Excellent":
                    return "Premium_Engaged"
                elif financial_cat == "Stable":
                    return "Growth_Focused"
                else:
                    return "Recovery_Engaged"
            elif engagement > 0.25:
                if financial_cat == "Excellent":
                    return "Premium_Moderate"
                elif financial_cat == "Stable":
                    return "Mainstream"
                else:
                    return "Recovery_Moderate"
            else:
                if financial_cat == "At_Risk":
                    return "Financial_Priority"
                else:
                    return "Activation_Needed"
        
        self.user_features['enhanced_segment'] = self.user_features.apply(assign_segment_name, axis=1)
        
        print("Enhanced segment distribution:")
        print(self.user_features['enhanced_segment'].value_counts())
        
        return self.user_features
    
    def generate_financial_content_recommendations(self):
        """Generate financially-aware content recommendations"""
        print("Generating financially-aware recommendations...")
        
        recommendations = []
        content_types = ['improve', 'insights', 'drivescore', 'protect', 'credit_cards', 'loans']
        
        for user_id, user_data in self.user_features.iterrows():
            segment = user_data['enhanced_segment']
            financial_cat = user_data['financial_category']
            credit_score = user_data['credit_score']
            dti_ratio = user_data['dti_ratio']
            has_ccj = user_data['has_ccj']
            missed_payments = user_data['missed_payments']
            
            # Financial priority recommendations
            financial_priorities = self.get_financial_priorities(user_data)
            
            # Base content preferences
            user_prefs = {}
            for content_type in content_types:
                user_prefs[content_type] = user_data.get(f'pref_{content_type}', 0)
            
            # Apply financial context to content scoring
            content_scores = {}
            
            for content_type in content_types:
                base_score = user_prefs[content_type]
                
                # Apply financial relevance multipliers
                if content_type == 'improve' and credit_score < 650:
                    base_score *= 2.0  # Credit improvement is high priority
                elif content_type == 'protect' and financial_cat == "Excellent":
                    base_score *= 1.5  # Wealth protection for financially healthy users
                elif content_type == 'loans' and dti_ratio > 0.6:
                    base_score *= 0.5  # Reduce loan recommendations for high DTI
                elif content_type == 'credit_cards' and has_ccj:
                    base_score *= 0.3  # Reduce credit card recommendations for CCJ users
                elif content_type == 'drivescore' and financial_cat == "At_Risk":
                    base_score *= 1.8  # Financial education priority for at-risk users
                elif content_type == 'insights' and missed_payments > 2:
                    base_score *= 1.7  # Financial insights for users with payment issues
                
                content_scores[content_type] = base_score
            
            # Sort by adjusted scores
            sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Generate segment-specific strategies
            strategy = self.get_strategy_by_segment(segment, user_data)
            
            # Add urgency flags
            urgency_flags = self.get_urgency_flags(user_data)
            
            recommendations.append({
                'user_id': user_id,
                'enhanced_segment': segment,
                'financial_category': financial_cat,
                'primary_recommendation': sorted_content[0][0],
                'secondary_recommendation': sorted_content[1][0],
                'tertiary_recommendation': sorted_content[2][0],
                'financial_priorities': ', '.join(financial_priorities),
                'urgency_flags': ', '.join(urgency_flags),
                'strategy': strategy,
                'credit_score': credit_score,
                'dti_ratio': dti_ratio,
                'financial_health_score': user_data['financial_health_score'],
                'engagement_score': user_data['engagement_score']
            })
        
        return pd.DataFrame(recommendations)
    
    def get_financial_priorities(self, user_data):
        """Identify financial priorities for a user"""
        priorities = []
        
        if user_data['credit_score'] < 650:
            priorities.append("Credit_Repair")
        if user_data['dti_ratio'] > 0.6:
            priorities.append("Debt_Reduction")
        if user_data['missed_payments'] > 2:
            priorities.append("Payment_Management")
        if user_data['has_ccj']:
            priorities.append("Legal_Financial_Issues")
        if user_data['total_debt'] > user_data['income'] * 0.8:
            priorities.append("Debt_Consolidation")
        if user_data['financial_health_score'] > 0.7 and user_data['income'] > 50000:
            priorities.append("Wealth_Building")
        if not user_data['has_mortgage'] and user_data['financial_health_score'] > 0.6:
            priorities.append("Homeownership_Ready")
        
        return priorities if priorities else ["General_Financial_Wellness"]
    
    def get_urgency_flags(self, user_data):
        """Identify urgent financial issues"""
        flags = []
        
        if user_data['dti_ratio'] > 0.8:
            flags.append("HIGH_DEBT_BURDEN")
        if user_data['credit_score'] < 500:
            flags.append("CRITICAL_CREDIT_SCORE")
        if user_data['missed_payments'] > 4:
            flags.append("PAYMENT_CRISIS")
        if user_data['has_ccj']:
            flags.append("LEGAL_ACTION")
        
        return flags if flags else ["STABLE"]
    
    def get_strategy_by_segment(self, segment, user_data):
        """Get tailored strategy based on enhanced segment"""
        strategies = {
            "Premium_Engaged": f"Offer premium wealth management and investment content. Focus on portfolio optimization and advanced financial strategies. Credit score: {user_data['credit_score']}.",
            
            "Growth_Focused": f"Provide growth-oriented financial content with moderate complexity. Focus on building wealth and improving financial position. Current DTI: {user_data['dti_ratio']:.2f}.",
            
            "Recovery_Engaged": f"Deliver financial recovery content with high engagement. Focus on debt management and credit repair while maintaining engagement. Priority: Credit improvement from {user_data['credit_score']}.",
            
            "Premium_Moderate": f"Offer premium content with clear value propositions. Balance wealth building with practical financial advice. Leverage high financial health score: {user_data['financial_health_score']:.2f}.",
            
            "Mainstream": f"Provide balanced financial content for stable users. Focus on practical advice and gradual improvement. Build on stable financial foundation.",
            
            "Recovery_Moderate": f"Deliver accessible financial recovery content. Simplify complex concepts and focus on immediate actionable steps. Address DTI ratio: {user_data['dti_ratio']:.2f}.",
            
            "Financial_Priority": f"Urgent: Focus on critical financial issues first. Provide crisis management content and immediate help resources. Address multiple risk factors.",
            
            "Activation_Needed": f"Basic financial education and engagement building. Start with simple concepts and gradually increase complexity. Build financial awareness."
        }
        
        return strategies.get(segment, "Provide general financial guidance based on user profile.")
    
    def print_enhanced_recommendations(self, recommendations_df, n_samples=10):
        """Print sample enhanced recommendations"""
        print("\n" + "=" * 80)
        print("ENHANCED FINANCIALLY-AWARE RECOMMENDATIONS")
        print("=" * 80)
        
        for _, user in recommendations_df.head(n_samples).iterrows():
            print(f"\nUser: {user['user_id']}")
            print(f"Enhanced Segment: {user['enhanced_segment']}")
            print(f"Financial Category: {user['financial_category']}")
            print(f"Credit Score: {user['credit_score']} | DTI: {user['dti_ratio']:.2f} | Health Score: {user['financial_health_score']:.2f}")
            print(f"Primary Rec: {user['primary_recommendation']}")
            print(f"Financial Priorities: {user['financial_priorities']}")
            print(f"Urgency Flags: {user['urgency_flags']}")
            print(f"Strategy: {user['strategy'][:100]}...")
            print("-" * 80)
    
    def analyze_segments_by_financial_health(self):
        """Analyze the relationship between segments and financial health"""
        print("\n" + "=" * 60)
        print("SEGMENT ANALYSIS BY FINANCIAL HEALTH")
        print("=" * 60)
        
        segment_analysis = self.user_features.groupby('enhanced_segment').agg({
            'financial_health_score': ['mean', 'std'],
            'credit_score': ['mean', 'min', 'max'],
            'dti_ratio': ['mean', 'std'],
            'income': ['mean', 'median'],
            'engagement_score': ['mean']
        }).round(2)
        
        print(segment_analysis)
        
        return segment_analysis
    
    def run_enhanced_analysis(self):
        """Run the complete enhanced analysis"""
        print("Starting Enhanced Financially-Aware Recommendation Analysis...")
        
        # Create enhanced features
        self.create_enhanced_user_features()
        
        # Perform enhanced segmentation
        self.perform_enhanced_segmentation()
        
        # Generate recommendations
        recommendations = self.generate_financial_content_recommendations()
        
        # Print sample recommendations
        self.print_enhanced_recommendations(recommendations)
        
        # Analyze segments
        self.analyze_segments_by_financial_health()
        
        return recommendations

if __name__ == "__main__":
    # Initialize the enhanced recommender
    recommender = FinanciallyAwareRecommender('joined_user_table.csv')
    
    # Run enhanced analysis
    enhanced_recommendations = recommender.run_enhanced_analysis()
    
    # Save enhanced recommendations
    enhanced_recommendations.to_csv('enhanced_financial_recommendations.csv', index=False)
    print(f"\nEnhanced recommendations saved to 'enhanced_financial_recommendations.csv'")
    
    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("=" * 60) 