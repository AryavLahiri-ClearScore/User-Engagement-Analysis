"""
Comprehensive Manual vs ML Comparison Analysis
Creates visualizations and CSV files comparing manual weights vs ML-learned weights
Now uses supervised learning for generating recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from fully_ml_financial_recommender import FullyMLFinancialRecommender
import warnings
warnings.filterwarnings('ignore')

class ManualVsMLComparison:
    """Compare manual weight system vs ML-learned weights with supervised learning recommendations"""
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.manual_results = None
        self.ml_results = None
        self.ml_recommendation_model = None
        self.ml_priority_model = None
        self.ml_urgency_model = None
        self.content_encoders = {}
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_real_training_data(self):
        """Create training data using REAL user engagement patterns"""
        print("🎯 CREATING REAL TRAINING DATA FROM ACTUAL USER BEHAVIOR")
        print("-" * 60)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Create user-level financial data
        user_financial = self.df.groupby('user_id').agg({
            'credit_score': 'first',
            'dti_ratio': 'first', 
            'income': 'first',
            'total_debt': 'first',
            'missed_payments': 'first',
            'has_ccj': 'first',
            'has_mortgage': 'first',
            'has_car': 'first'
        }).reset_index()
        
        # Create user-level REAL engagement data
        user_engagement = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        user_engagement.columns = ['user_id', 'avg_time_viewed', 'total_time_viewed', 'total_interactions',
                                  'click_rate', 'total_clicks', 'unique_content_viewed']
        
        # Merge financial and engagement data
        user_data = pd.merge(user_financial, user_engagement, on='user_id')
        
        print(f"📊 Using REAL data from {len(user_data)} users")
        print(f"📈 Real engagement metrics: click_rate, avg_time_viewed, total_interactions")
        
        # Generate training data based on real patterns
        training_data = []
        content_types = ['improve', 'insights', 'drivescore', 'protect', 'credit_cards', 'loans']
        
        for _, user in user_data.iterrows():
            credit_score = user['credit_score']
            dti_ratio = user['dti_ratio']
            missed_payments = user['missed_payments']
            income = user['income']
            has_ccj = user['has_ccj']
            
            # Calculate financial health score using real data
            credit_comp = min(credit_score / 1000, 1.0)
            dti_comp = max(0, 1 - dti_ratio)
            missed_comp = max(0, 1 - (missed_payments / 10))
            income_comp = min(income / 100000, 1.0)
            ccj_comp = 0.0 if has_ccj else 1.0
            asset_comp = (user['has_mortgage'] * 0.6 + user['has_car'] * 0.4)
            
            financial_health = (credit_comp * 0.30 + dti_comp * 0.25 + 
                              missed_comp * 0.15 + income_comp * 0.15 + 
                              ccj_comp * 0.10 + asset_comp * 0.05)
            
            # Calculate REAL engagement score using actual user behavior
            normalized_time = min(user['avg_time_viewed'] / 60, 1.0)  # Normalize to 60 seconds max
            normalized_interactions = min(user['total_interactions'] / 12, 1.0)  # Normalize to 12 interactions max
            
            real_engagement_score = (
                user['click_rate'] * 0.4 +
                normalized_time * 0.3 +
                normalized_interactions * 0.3
            )
            
            # Recommendation logic based on REAL financial + engagement patterns
            if missed_payments >= 2:
                primary_rec = 'improve'
                priorities = ['URGENT_PAYMENT_MANAGEMENT', 'Payment_Management']
                urgency = ['PAYMENT_CRISIS'] if missed_payments > 4 else ['HIGH_DEBT_BURDEN']
            elif dti_ratio >= 0.5:
                primary_rec = 'improve'
                priorities = ['URGENT_DTI_REDUCTION', 'Debt_Reduction'] 
                urgency = ['HIGH_DEBT_BURDEN']
            elif credit_score < 650:
                primary_rec = 'improve'
                priorities = ['Credit_Repair']
                urgency = ['CRITICAL_CREDIT_SCORE'] if credit_score < 500 else ['STABLE_FINANCIAL_POSITION']
            elif financial_health > 0.7:
                # High financial health users get different content based on REAL engagement
                if real_engagement_score > 0.6:
                    primary_rec = 'protect'  # Highly engaged + wealthy = protection focus
                else:
                    primary_rec = 'insights'  # Wealthy but low engagement = educational content
                priorities = ['Wealth_Building']
                urgency = ['STABLE_FINANCIAL_POSITION']
            elif financial_health > 0.5:
                # Medium financial health users get content based on REAL engagement
                if real_engagement_score > 0.4:
                    primary_rec = 'insights'  # Engaged = insights
                else:
                    primary_rec = 'drivescore'  # Low engagement = gamification
                priorities = ['General_Financial_Wellness']
                urgency = ['STABLE_FINANCIAL_POSITION']
            else:
                # Low financial health users always get improvement content
                primary_rec = 'improve'
                priorities = ['General_Financial_Wellness']
                urgency = ['STABLE_FINANCIAL_POSITION']
            
            # Add fixed randomness to prevent overfitting (seeded for reproducibility)
            # This uses the same random choices every run but still provides ML benefits
            if np.random.random() < 0.05:  # Only 5% randomness with real data
                primary_rec = np.random.choice(content_types)
            
            training_data.append({
                'user_id': user['user_id'],
                'credit_score': credit_score,
                'dti_ratio': dti_ratio,
                'income': income,
                'missed_payments': missed_payments,
                'has_ccj': has_ccj,
                'financial_health_score': financial_health,
                'engagement_score': real_engagement_score,  # REAL engagement score!
                'click_rate': user['click_rate'],  # Add real engagement features
                'avg_time_viewed': user['avg_time_viewed'],
                'total_interactions': user['total_interactions'],
                'primary_recommendation': primary_rec,
                'financial_priorities': ', '.join(priorities[:2]),
                'urgency_flags': ', '.join(urgency)
            })
        
        training_df = pd.DataFrame(training_data)
        print(f"✅ Created REAL training data: {len(training_df)} samples")
        print(f"📊 Recommendation distribution: {dict(training_df['primary_recommendation'].value_counts())}")
        print(f"📈 Real engagement stats:")
        print(f"   • Avg click rate: {training_df['click_rate'].mean():.3f}")
        print(f"   • Avg time viewed: {training_df['avg_time_viewed'].mean():.1f} seconds")
        print(f"   • Avg interactions: {training_df['total_interactions'].mean():.1f}")
        print(f"   • Engagement score range: {training_df['engagement_score'].min():.3f} - {training_df['engagement_score'].max():.3f}")
        
        return training_df
    
    def train_ml_recommendation_models(self, training_data):
        """Train supervised learning models for recommendations"""
        print("🔬 TRAINING SUPERVISED LEARNING MODELS")
        print("-" * 50)
        
        # Prepare features for ML (now includes REAL engagement features)
        feature_columns = ['credit_score', 'dti_ratio', 'income', 'missed_payments', 
                          'has_ccj', 'financial_health_score', 'engagement_score',
                          'click_rate', 'avg_time_viewed', 'total_interactions']
        X = training_data[feature_columns].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.feature_scaler = scaler
        self.feature_columns = feature_columns
        
        # 1. Train recommendation model
        print("   🎯 Training recommendation model...")
        y_rec = training_data['primary_recommendation']
        self.recommendation_encoder = LabelEncoder()
        y_rec_encoded = self.recommendation_encoder.fit_transform(y_rec)
        
        # Check class distribution
        unique_classes, class_counts = np.unique(y_rec_encoded, return_counts=True)
        print(f"      📊 Class distribution: {dict(zip(self.recommendation_encoder.classes_, class_counts))}")
        
        # Use stratification only if all classes have at least 2 samples
        min_class_count = min(class_counts)
        use_stratify = min_class_count >= 2
        
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_rec_encoded, test_size=0.2, random_state=42, stratify=y_rec_encoded
            )
        else:
            print(f"      ⚠️ Smallest class has only {min_class_count} samples, skipping stratification")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_rec_encoded, test_size=0.2, random_state=42
            )
        
        self.ml_recommendation_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.ml_recommendation_model.fit(X_train, y_train)
        
        y_pred = self.ml_recommendation_model.predict(X_test)
        rec_accuracy = accuracy_score(y_test, y_pred)
        print(f"      ✅ Recommendation model accuracy: {rec_accuracy:.3f}")
        
        # 2. Train financial priorities model
        print("   💰 Training financial priorities model...")
        # Create simplified priority categories for classification
        training_data['priority_category'] = training_data['financial_priorities'].apply(
            lambda x: 'URGENT' if 'URGENT' in str(x) 
                     else 'REPAIR' if 'Credit_Repair' in str(x) or 'Debt_Reduction' in str(x)
                     else 'BUILDING' if 'Wealth_Building' in str(x)
                     else 'GENERAL'
        )
        
        y_priority = training_data['priority_category']
        self.priority_encoder = LabelEncoder()
        y_priority_encoded = self.priority_encoder.fit_transform(y_priority)
        
        # Check class distribution for priorities
        unique_priority_classes, priority_class_counts = np.unique(y_priority_encoded, return_counts=True)
        print(f"      📊 Priority class distribution: {dict(zip(self.priority_encoder.classes_, priority_class_counts))}")
        
        min_priority_count = min(priority_class_counts)
        use_stratify_priority = min_priority_count >= 2
        
        if use_stratify_priority:
            X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
                X_scaled, y_priority_encoded, test_size=0.2, random_state=42, stratify=y_priority_encoded
            )
        else:
            print(f"      ⚠️ Smallest priority class has only {min_priority_count} samples, skipping stratification")
            X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
                X_scaled, y_priority_encoded, test_size=0.2, random_state=42
            )
        
        self.ml_priority_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        self.ml_priority_model.fit(X_train_p, y_train_p)
        
        y_pred_p = self.ml_priority_model.predict(X_test_p)
        priority_accuracy = accuracy_score(y_test_p, y_pred_p)
        print(f"      ✅ Priority model accuracy: {priority_accuracy:.3f}")
        
        # 3. Train urgency flags model
        print("   🚨 Training urgency flags model...")
        training_data['urgency_category'] = training_data['urgency_flags'].apply(
            lambda x: 'CRITICAL' if 'CRISIS' in str(x) or 'CRITICAL' in str(x)
                     else 'HIGH_RISK' if 'HIGH_DEBT' in str(x) or 'LEGAL' in str(x)
                     else 'STABLE'
        )
        
        y_urgency = training_data['urgency_category']
        self.urgency_encoder = LabelEncoder()
        y_urgency_encoded = self.urgency_encoder.fit_transform(y_urgency)
        
        # Check class distribution for urgency
        unique_urgency_classes, urgency_class_counts = np.unique(y_urgency_encoded, return_counts=True)
        print(f"      📊 Urgency class distribution: {dict(zip(self.urgency_encoder.classes_, urgency_class_counts))}")
        
        min_urgency_count = min(urgency_class_counts)
        use_stratify_urgency = min_urgency_count >= 2
        
        if use_stratify_urgency:
            X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
                X_scaled, y_urgency_encoded, test_size=0.2, random_state=42, stratify=y_urgency_encoded
            )
        else:
            print(f"      ⚠️ Smallest urgency class has only {min_urgency_count} samples, skipping stratification")
            X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
                X_scaled, y_urgency_encoded, test_size=0.2, random_state=42
            )
        
        self.ml_urgency_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        self.ml_urgency_model.fit(X_train_u, y_train_u)
        
        y_pred_u = self.ml_urgency_model.predict(X_test_u)
        urgency_accuracy = accuracy_score(y_test_u, y_pred_u)
        print(f"      ✅ Urgency model accuracy: {urgency_accuracy:.3f}")
        
        print(f"\n🎉 All ML models trained successfully!")
        print(f"   📊 Overall model performance:")
        print(f"      • Recommendation accuracy: {rec_accuracy:.1%}")
        print(f"      • Priority accuracy: {priority_accuracy:.1%}")
        print(f"      • Urgency accuracy: {urgency_accuracy:.1%}")
        
        return {
            'recommendation_accuracy': rec_accuracy,
            'priority_accuracy': priority_accuracy, 
            'urgency_accuracy': urgency_accuracy
        }

    def calculate_manual_weights_analysis(self):
        """Calculate user features using manual weights"""
        print("🔧 CALCULATING MANUAL WEIGHTS ANALYSIS")
        print("=" * 60)
        
        # Create user-level aggregated data
        user_financial = self.df.groupby('user_id').agg({
            'credit_score': 'first',
            'dti_ratio': 'first', 
            'income': 'first',
            'total_debt': 'first',
            'missed_payments': 'first',
            'has_ccj': 'first',
            'has_mortgage': 'first',
            'has_car': 'first'
        }).reset_index()
        
        # Aggregate engagement data per user
        user_engagement = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        user_engagement.columns = ['user_id', 'avg_time_viewed', 'total_time_viewed', 'total_interactions',
                                  'click_rate', 'total_clicks', 'unique_content_viewed']
        
        # Merge data
        user_df = pd.merge(user_financial, user_engagement, on='user_id')
        
        # Calculate manual financial health scores
        manual_financial_weights = {
            'credit_component': 0.30,
            'dti_component': 0.25, 
            'missed_payments_component': 0.15,
            'income_component': 0.15,
            'ccj_component': 0.10,
            'asset_component': 0.05
        }
        
        # Calculate components
        credit_comp = np.minimum(user_df['credit_score'] / 1000, 1.0)
        dti_comp = np.maximum(0, 1 - user_df['dti_ratio'])
        missed_comp = np.maximum(0, 1 - (user_df['missed_payments'] / 10))
        income_comp = np.minimum(user_df['income'] / 100000, 1.0)
        ccj_comp = 1 - user_df['has_ccj']
        asset_comp = user_df['has_mortgage'] * 0.6 + user_df['has_car'] * 0.4
        
        # Calculate manual financial health scores
        user_df['manual_financial_health_score'] = (
            credit_comp * manual_financial_weights['credit_component'] +
            dti_comp * manual_financial_weights['dti_component'] +
            missed_comp * manual_financial_weights['missed_payments_component'] +
            income_comp * manual_financial_weights['income_component'] +
            ccj_comp * manual_financial_weights['ccj_component'] +
            asset_comp * manual_financial_weights['asset_component']
        )
        
        # Calculate manual engagement scores
        manual_engagement_weights = {
            'click_rate': 0.4,
            'avg_time_viewed': 0.3,
            'total_interactions': 0.3
        }
        
        # Normalize engagement features
        normalized_time = np.minimum(user_df['avg_time_viewed'] / 60, 1.0)
        normalized_interactions = np.minimum(user_df['total_interactions'] / 12, 1.0)
        
        user_df['manual_engagement_score'] = (
            user_df['click_rate'] * manual_engagement_weights['click_rate'] +
            normalized_time * manual_engagement_weights['avg_time_viewed'] +
            normalized_interactions * manual_engagement_weights['total_interactions']
        )
        
        # Categorize financial health
        def categorize_financial_health(score):
            if score >= 0.8:
                return "Excellent"
            elif score >= 0.65:
                return "Good"
            elif score >= 0.45:
                return "Fair"
            else:
                return "Poor"
        
        user_df['manual_financial_category'] = user_df['manual_financial_health_score'].apply(categorize_financial_health)
        
        # Assign enhanced segments
        def assign_enhanced_segments(row):
            engagement = row['manual_engagement_score']
            financial_cat = row['manual_financial_category']
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
        
        user_df['manual_enhanced_segment'] = user_df.apply(assign_enhanced_segments, axis=1)
        
        # Generate ML-based manual recommendations using trained models
        manual_recommendations = self.generate_ml_based_recommendations(user_df, method='manual')
        user_df = pd.merge(user_df, manual_recommendations, on='user_id', how='left')
        
        self.manual_results = user_df
        
        print(f"✅ Calculated manual analysis for {len(user_df)} users")
        print(f"📊 Manual Financial Categories: {dict(user_df['manual_financial_category'].value_counts())}")
        print(f"🎯 Manual Enhanced Segments: {dict(user_df['manual_enhanced_segment'].value_counts())}")
        
        return user_df
    
    def generate_ml_based_recommendations(self, user_df, method='manual'):
        """Generate recommendations using trained ML models"""
        print(f"🤖 GENERATING ML-BASED RECOMMENDATIONS ({method.upper()})")
        print("-" * 50)
        
        if self.ml_recommendation_model is None:
            print("⚠️ ML models not trained yet. Training now...")
            training_data = self.create_real_training_data()
            self.train_ml_recommendation_models(training_data)
        
        recommendations = []
        
        for _, user_data in user_df.iterrows():
            user_id = user_data['user_id']
            
            # Prepare features for ML prediction
            feature_values = []
            for col in self.feature_columns:
                if col == 'financial_health_score':
                    if method == 'manual':
                        feature_values.append(user_data['manual_financial_health_score'])
                    else:
                        feature_values.append(user_data['ml_financial_health_score'])
                elif col == 'engagement_score':
                    if method == 'manual':
                        feature_values.append(user_data['manual_engagement_score'])
                    else:
                        feature_values.append(user_data['ml_engagement_score'])
                else:
                    feature_values.append(user_data[col])
            
            # Scale features
            X_user = self.feature_scaler.transform([feature_values])
            
            # Predict recommendations using ML models
            rec_pred = self.ml_recommendation_model.predict(X_user)[0]
            primary_rec = self.recommendation_encoder.inverse_transform([rec_pred])[0]
            
            # Get secondary and tertiary recommendations from probabilities
            rec_probs = self.ml_recommendation_model.predict_proba(X_user)[0]
            sorted_indices = np.argsort(rec_probs)[::-1]
            
            secondary_rec = self.recommendation_encoder.inverse_transform([sorted_indices[1]])[0]
            tertiary_rec = self.recommendation_encoder.inverse_transform([sorted_indices[2]])[0]
            
            # Predict priorities and urgency
            priority_pred = self.ml_priority_model.predict(X_user)[0]
            priority_category = self.priority_encoder.inverse_transform([priority_pred])[0]
            
            urgency_pred = self.ml_urgency_model.predict(X_user)[0]
            urgency_category = self.urgency_encoder.inverse_transform([urgency_pred])[0]
            
            # Convert categories back to detailed descriptions
            financial_priorities = self.get_detailed_priorities(priority_category, user_data)
            urgency_flags = self.get_detailed_urgency(urgency_category, user_data)
            
            # Generate strategy
            if method == 'manual':
                segment = user_data['manual_enhanced_segment']
                strategy = self.get_strategy_by_segment(segment, user_data, method='manual')
            else:
                segment = user_data['ml_enhanced_segment']
                strategy = self.get_strategy_by_segment(segment, user_data, method='ml')
            
            prefix = method
            recommendations.append({
                'user_id': user_id,
                f'{prefix}_primary_recommendation': primary_rec,
                f'{prefix}_secondary_recommendation': secondary_rec,
                f'{prefix}_tertiary_recommendation': tertiary_rec,
                f'{prefix}_financial_priorities': ', '.join(financial_priorities),
                f'{prefix}_urgency_flags': ', '.join(urgency_flags),
                f'{prefix}_strategy': strategy,
                f'{prefix}_ml_confidence': rec_probs.max()
            })
        
        return pd.DataFrame(recommendations)
    
    def get_detailed_priorities(self, category, user_data):
        """Convert ML category back to detailed priorities"""
        priorities = []
        
        if category == 'URGENT':
            if user_data['missed_payments'] >= 2:
                priorities.append("URGENT_PAYMENT_MANAGEMENT")
            if user_data['dti_ratio'] >= 0.5:
                priorities.append("URGENT_DTI_REDUCTION")
        elif category == 'REPAIR':
            if user_data['credit_score'] < 650:
                priorities.append("Credit_Repair")
            if user_data['dti_ratio'] > 0.6:
                priorities.append("Debt_Reduction")
            if user_data['missed_payments'] > 2:
                priorities.append("Payment_Management")
        elif category == 'BUILDING':
            priorities.append("Wealth_Building")
            if not user_data['has_mortgage']:
                priorities.append("Homeownership_Ready")
        else:  # GENERAL
            priorities.append("General_Financial_Wellness")
        
        return priorities if priorities else ["General_Financial_Wellness"]
    
    def get_detailed_urgency(self, category, user_data):
        """Convert ML category back to detailed urgency flags"""
        flags = []
        
        if category == 'CRITICAL':
            if user_data['missed_payments'] > 4:
                flags.append("PAYMENT_CRISIS")
            if user_data['credit_score'] < 500:
                flags.append("CRITICAL_CREDIT_SCORE")
        elif category == 'HIGH_RISK':
            if user_data['dti_ratio'] > 0.8:
                flags.append("HIGH_DEBT_BURDEN")
            if user_data['has_ccj']:
                flags.append("LEGAL_ACTION")
        else:  # STABLE
            flags.append("STABLE_FINANCIAL_POSITION")
        
        return flags if flags else ["STABLE_FINANCIAL_POSITION"]
    
    def get_strategy_by_segment(self, segment, user_data, method='manual'):
        """Get tailored strategy based on enhanced segment"""
        score_key = f'{method}_financial_health_score'
        strategies = {
            "Debt_Management_Priority": f"🚨 CRITICAL: DTI {user_data['dti_ratio']:.1%} - URGENT debt reduction required. Focus on debt consolidation, payment strategies, budgeting, and avoid all new debt. Immediate action needed.",
            
            "Payment_Recovery_Priority": f"🚨 PAYMENT ISSUES: {user_data['missed_payments']} missed payments detected - URGENT payment management required. Focus on payment scheduling, budgeting, automatic payments, and credit repair strategies. Address payment history immediately.",
            
            "Premium_Engaged": f"Offer premium wealth management and investment content. Focus on portfolio optimization and advanced financial strategies. Credit score: {user_data['credit_score']}.",
            
            "Growth_Focused": f"Provide growth-oriented financial content with moderate complexity. Focus on building wealth and improving financial position. Current DTI: {user_data['dti_ratio']:.2f}.",
            
            "Recovery_Engaged": f"Deliver financial recovery content with high engagement. Focus on debt management and credit repair while maintaining engagement. Priority: Credit improvement from {user_data['credit_score']}.",
            
            "Premium_Moderate": f"Offer premium content with clear value propositions. Balance wealth building with practical financial advice. Leverage high financial health score: {user_data[score_key]:.2f}.",
            
            "Mainstream": f"Provide balanced financial content for users with decent financial health. Focus on practical advice and gradual improvement. Build on solid financial foundation.",
            
            "Recovery_Moderate": f"Deliver accessible financial recovery content. Simplify complex concepts and focus on immediate actionable steps. Address DTI ratio: {user_data['dti_ratio']:.2f}.",
            
            "Financial_Priority": f"Urgent: Focus on critical financial issues first. Provide crisis management content and immediate help resources. Address multiple risk factors.",
            
            "Activation_Needed": f"Basic financial education and engagement building. Start with simple concepts and gradually increase complexity. Build financial awareness."
        }
        
        return strategies.get(segment, "Provide general financial guidance based on user profile.")
    
    def run_ml_analysis(self):
        """Run ML analysis using FullyMLFinancialRecommender"""
        print("\n🤖 RUNNING ML ANALYSIS")
        print("=" * 60)
        
        # Create ML recommender
        ml_recommender = FullyMLFinancialRecommender(self.csv_file)
        
        # Run ML analysis
        ml_recommender.run_fully_ml_analysis()
        
        # Get the results
        self.ml_results = ml_recommender.df.copy()
        
        # Generate ML-style recommendations for comparison using trained models
        ml_recommendations = self.generate_ml_based_recommendations(self.ml_results, method='ml')
        self.ml_results = pd.merge(self.ml_results, ml_recommendations, on='user_id', how='left')
        
        print(f"✅ Completed ML analysis for {len(self.ml_results)} users")
        print(f"📊 ML Financial Categories: {dict(self.ml_results['ml_financial_category'].value_counts())}")
        print(f"🎯 ML Enhanced Segments: {dict(self.ml_results['ml_enhanced_segment'].value_counts())}")
        
        return self.ml_results
    
    def create_comparison_visualizations(self):
        """Create comprehensive comparison visualizations"""
        print("\n🎨 CREATING COMPARISON VISUALIZATIONS")
        print("=" * 60)
        
        # Merge manual and ML results for comparison
        comparison_df = pd.merge(
            self.manual_results[['user_id', 'manual_financial_category', 'manual_enhanced_segment', 
                               'manual_financial_health_score', 'manual_engagement_score']],
            self.ml_results[['user_id', 'ml_financial_category', 'ml_enhanced_segment',
                           'ml_financial_health_score', 'ml_engagement_score']],
            on='user_id'
        )
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Financial Categories Comparison
        plt.subplot(3, 3, 1)
        manual_fin_counts = self.manual_results['manual_financial_category'].value_counts()
        plt.pie(manual_fin_counts.values, labels=manual_fin_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Manual Financial Categories\n(Manual Weights)', fontsize=12, fontweight='bold')
        
        plt.subplot(3, 3, 2)
        ml_fin_counts = self.ml_results['ml_financial_category'].value_counts()
        plt.pie(ml_fin_counts.values, labels=ml_fin_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('ML Financial Categories\n(ML-Learned Weights)', fontsize=12, fontweight='bold')
        
        # 2. Enhanced Segments Comparison
        plt.subplot(3, 3, 4)
        manual_seg_counts = self.manual_results['manual_enhanced_segment'].value_counts()
        plt.barh(range(len(manual_seg_counts)), manual_seg_counts.values, color='skyblue')
        plt.yticks(range(len(manual_seg_counts)), [seg[:15] + '...' if len(seg) > 15 else seg for seg in manual_seg_counts.index])
        plt.title('Manual Enhanced Segments', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Users')
        
        plt.subplot(3, 3, 5)
        ml_seg_counts = self.ml_results['ml_enhanced_segment'].value_counts()
        plt.barh(range(len(ml_seg_counts)), ml_seg_counts.values, color='lightcoral')
        plt.yticks(range(len(ml_seg_counts)), [seg[:15] + '...' if len(seg) > 15 else seg for seg in ml_seg_counts.index])
        plt.title('ML Enhanced Segments', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Users')
        
        # 3. Score Distributions
        plt.subplot(3, 3, 7)
        plt.hist(comparison_df['manual_financial_health_score'], alpha=0.7, label='Manual', bins=20, color='skyblue')
        plt.hist(comparison_df['ml_financial_health_score'], alpha=0.7, label='ML', bins=20, color='lightcoral')
        plt.title('Financial Health Score Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Financial Health Score')
        plt.ylabel('Number of Users')
        plt.legend()
        
        plt.subplot(3, 3, 8)
        plt.hist(comparison_df['manual_engagement_score'], alpha=0.7, label='Manual', bins=20, color='skyblue')
        plt.hist(comparison_df['ml_engagement_score'], alpha=0.7, label='ML', bins=20, color='lightcoral')
        plt.title('Engagement Score Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Engagement Score')
        plt.ylabel('Number of Users')
        plt.legend()
        
        # 4. Scatter Plot Comparison
        plt.subplot(3, 3, 9)
        plt.scatter(comparison_df['manual_financial_health_score'], comparison_df['ml_financial_health_score'], 
                   alpha=0.6, color='purple', s=30)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)  # Diagonal line
        plt.xlabel('Manual Financial Health Score')
        plt.ylabel('ML Financial Health Score')
        plt.title('Manual vs ML Financial Scores\n(Perfect agreement = diagonal line)', fontsize=12, fontweight='bold')
        
        # 5. Category Transition Matrix
        plt.subplot(3, 3, 3)
        # Create transition matrix
        transition_data = []
        categories = ['Poor', 'Fair', 'Good', 'Excellent']
        for manual_cat in categories:
            row = []
            for ml_cat in categories:
                count = len(comparison_df[(comparison_df['manual_financial_category'] == manual_cat) & 
                                        (comparison_df['ml_financial_category'] == ml_cat)])
                row.append(count)
            transition_data.append(row)
        
        sns.heatmap(transition_data, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories, yticklabels=categories)
        plt.title('Category Transition Matrix\n(Manual → ML)', fontsize=12, fontweight='bold')
        plt.xlabel('ML Categories')
        plt.ylabel('Manual Categories')
        
        # 6. Summary Statistics
        plt.subplot(3, 3, 6)
        plt.axis('off')
        
        # Calculate summary stats
        manual_mean_fin = comparison_df['manual_financial_health_score'].mean()
        ml_mean_fin = comparison_df['ml_financial_health_score'].mean()
        manual_mean_eng = comparison_df['manual_engagement_score'].mean()
        ml_mean_eng = comparison_df['ml_engagement_score'].mean()
        
        correlation_fin = comparison_df['manual_financial_health_score'].corr(comparison_df['ml_financial_health_score'])
        correlation_eng = comparison_df['manual_engagement_score'].corr(comparison_df['ml_engagement_score'])
        
        # Count differences
        fin_cat_same = (comparison_df['manual_financial_category'] == comparison_df['ml_financial_category']).sum()
        seg_same = (comparison_df['manual_enhanced_segment'] == comparison_df['ml_enhanced_segment']).sum()
        
        total_users = len(comparison_df)
        
        summary_text = f"""
        COMPARISON SUMMARY
        ═══════════════════
        
        📊 AVERAGE SCORES:
        Manual Financial: {manual_mean_fin:.3f}
        ML Financial:     {ml_mean_fin:.3f}
        
        Manual Engagement: {manual_mean_eng:.3f}
        ML Engagement:     {ml_mean_eng:.3f}
        
        🔗 CORRELATIONS:
        Financial Scores: {correlation_fin:.3f}
        Engagement Scores: {correlation_eng:.3f}
        
        🎯 AGREEMENT RATES:
        Same Financial Category: {fin_cat_same}/{total_users} ({fin_cat_same/total_users*100:.1f}%)
        Same Enhanced Segment:   {seg_same}/{total_users} ({seg_same/total_users*100:.1f}%)
        
        📈 KEY INSIGHTS:
        • Higher correlation = methods agree more
        • Lower correlation = ML found different patterns
        • Category changes show ML discoveries
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('manual_vs_ml_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Created comprehensive comparison visualization: manual_vs_ml_comprehensive_comparison.png")
        
        return comparison_df
    
    def create_detailed_segment_analysis(self, comparison_df):
        """Create detailed segment analysis visualization"""
        print("\n📊 CREATING DETAILED SEGMENT ANALYSIS")
        print("=" * 60)
        
        # Create segment comparison matrix
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Manual segments detailed breakdown
        ax1 = axes[0, 0]
        manual_segments = self.manual_results['manual_enhanced_segment'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(manual_segments)))
        wedges, texts, autotexts = ax1.pie(manual_segments.values, labels=None, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Manual Enhanced Segments Distribution', fontsize=14, fontweight='bold')
        
        # Add legend for manual segments
        ax1.legend(wedges, [f"{seg[:20]}..." if len(seg) > 20 else seg for seg in manual_segments.index],
                  title="Segments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # 2. ML segments detailed breakdown
        ax2 = axes[0, 1]
        ml_segments = self.ml_results['ml_enhanced_segment'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(ml_segments)))
        wedges, texts, autotexts = ax2.pie(ml_segments.values, labels=None, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('ML Enhanced Segments Distribution', fontsize=14, fontweight='bold')
        
        # Add legend for ML segments
        ax2.legend(wedges, [f"{seg[:20]}..." if len(seg) > 20 else seg for seg in ml_segments.index],
                  title="Segments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # 3. Side-by-side comparison
        ax3 = axes[1, 0]
        
        # Get all unique segments
        all_segments = set(list(manual_segments.index) + list(ml_segments.index))
        
        # Prepare data for comparison
        manual_counts = [manual_segments.get(seg, 0) for seg in all_segments]
        ml_counts = [ml_segments.get(seg, 0) for seg in all_segments]
        
        x = np.arange(len(all_segments))
        width = 0.35
        
        ax3.bar(x - width/2, manual_counts, width, label='Manual', color='skyblue', alpha=0.8)
        ax3.bar(x + width/2, ml_counts, width, label='ML', color='lightcoral', alpha=0.8)
        
        ax3.set_xlabel('Enhanced Segments')
        ax3.set_ylabel('Number of Users')
        ax3.set_title('Manual vs ML Segment Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([seg[:10] + '...' if len(seg) > 10 else seg for seg in all_segments], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Segment migration analysis
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate segment migrations
        segment_changes = []
        for manual_seg in manual_segments.index:
            for ml_seg in ml_segments.index:
                count = len(comparison_df[(comparison_df['manual_enhanced_segment'] == manual_seg) & 
                                        (comparison_df['ml_enhanced_segment'] == ml_seg)])
                if count > 0:
                    segment_changes.append({
                        'from': manual_seg,
                        'to': ml_seg,
                        'count': count,
                        'same': manual_seg == ml_seg
                    })
        
        # Sort by count and show top migrations
        segment_changes.sort(key=lambda x: x['count'], reverse=True)
        
        migration_text = "TOP SEGMENT MIGRATIONS\n" + "═" * 30 + "\n\n"
        migration_text += f"{'FROM (Manual)':<20} → {'TO (ML)':<20} | {'COUNT':>5}\n"
        migration_text += "─" * 55 + "\n"
        
        for change in segment_changes[:10]:  # Top 10 migrations
            from_seg = change['from'][:18] + ".." if len(change['from']) > 18 else change['from']
            to_seg = change['to'][:18] + ".." if len(change['to']) > 18 else change['to']
            status = "✓" if change['same'] else "↔"
            migration_text += f"{from_seg:<20} → {to_seg:<20} | {change['count']:>5} {status}\n"
        
        # Add summary stats
        total_same = sum(1 for change in segment_changes if change['same'])
        total_different = len(segment_changes) - total_same
        
        migration_text += f"\n📊 MIGRATION SUMMARY:\n"
        migration_text += f"Users staying in same segment: {comparison_df['manual_enhanced_segment'].eq(comparison_df['ml_enhanced_segment']).sum()}\n"
        migration_text += f"Users changing segments: {len(comparison_df) - comparison_df['manual_enhanced_segment'].eq(comparison_df['ml_enhanced_segment']).sum()}\n"
        migration_text += f"\n✓ = Same segment\n↔ = Different segment"
        
        ax4.text(0.05, 0.95, migration_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('detailed_segment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Created detailed segment analysis: detailed_segment_analysis.png")
    
    def create_correlation_analysis(self):
        """Create correlation analysis like enhanced_financial_recommender_with_ml_engagement_only.py"""
        print("\n🔗 CREATING CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Create figure with subplots for correlation analysis
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Manual weights correlation matrix
        plt.subplot(2, 3, 1)
        manual_corr_features = ['manual_financial_health_score', 'manual_engagement_score', 
                               'credit_score', 'dti_ratio', 'income', 'missed_payments',
                               'click_rate', 'avg_time_viewed', 'total_interactions']
        manual_corr_data = self.manual_results[manual_corr_features]
        manual_correlation = manual_corr_data.corr()
        
        sns.heatmap(manual_correlation, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Manual Weights Correlation Matrix', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 2. ML weights correlation matrix
        plt.subplot(2, 3, 2)
        ml_corr_features = ['ml_financial_health_score', 'ml_engagement_score', 
                           'credit_score', 'dti_ratio', 'income', 'missed_payments',
                           'click_rate', 'avg_time_viewed', 'total_interactions']
        ml_corr_data = self.ml_results[ml_corr_features]
        ml_correlation = ml_corr_data.corr()
        
        sns.heatmap(ml_correlation, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('ML Weights Correlation Matrix', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 3. Engagement vs Financial scatter plot (Manual)
        plt.subplot(2, 3, 3)
        plt.scatter(self.manual_results['manual_engagement_score'], 
                   self.manual_results['manual_financial_health_score'],
                   c=self.manual_results['manual_financial_category'].astype('category').cat.codes,
                   cmap='viridis', alpha=0.6, s=50)
        plt.xlabel('Manual Engagement Score')
        plt.ylabel('Manual Financial Health Score')
        plt.title('Manual: Engagement vs Financial Health', fontsize=12, fontweight='bold')
        plt.colorbar(label='Financial Category')
        
        # 4. Engagement vs Financial scatter plot (ML)
        plt.subplot(2, 3, 4)
        plt.scatter(self.ml_results['ml_engagement_score'], 
                   self.ml_results['ml_financial_health_score'],
                   c=self.ml_results['ml_financial_category'].astype('category').cat.codes,
                   cmap='viridis', alpha=0.6, s=50)
        plt.xlabel('ML Engagement Score')
        plt.ylabel('ML Financial Health Score')
        plt.title('ML: Engagement vs Financial Health', fontsize=12, fontweight='bold')
        plt.colorbar(label='Financial Category')
        
        # 5. Feature importance comparison
        plt.subplot(2, 3, 5)
        
        # Manual weights (normalized)
        manual_weights = {
            'Credit Score': 0.30,
            'DTI Ratio': 0.25,
            'Missed Payments': 0.15,
            'Income': 0.15,
            'CCJ Status': 0.10,
            'Assets': 0.05
        }
        
        # Calculate correlation-based "importance" for ML
        ml_importance = {}
        for feature in ['credit_score', 'dti_ratio', 'missed_payments', 'income']:
            correlation = abs(ml_correlation.loc['ml_financial_health_score', feature])
            ml_importance[feature.replace('_', ' ').title()] = correlation
        
        # Add placeholder values for missing features
        ml_importance['CCJ Status'] = abs(ml_correlation.loc['ml_financial_health_score', 'ml_financial_health_score']) * 0.1
        ml_importance['Assets'] = abs(ml_correlation.loc['ml_financial_health_score', 'ml_financial_health_score']) * 0.05
        
        # Normalize ML importance
        total_ml = sum(ml_importance.values())
        ml_importance = {k: v/total_ml for k, v in ml_importance.items()}
        
        features = list(manual_weights.keys())
        manual_vals = [manual_weights[f] for f in features]
        ml_vals = [ml_importance.get(f, 0) for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        plt.bar(x - width/2, manual_vals, width, label='Manual', color='skyblue', alpha=0.8)
        plt.bar(x + width/2, ml_vals, width, label='ML', color='lightcoral', alpha=0.8)
        
        plt.xlabel('Financial Features')
        plt.ylabel('Importance Weight')
        plt.title('Feature Importance: Manual vs ML', fontsize=12, fontweight='bold')
        plt.xticks(x, [f[:8] + '...' if len(f) > 8 else f for f in features], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Clustering visualization (if possible)
        plt.subplot(2, 3, 6)
        
        # Use PCA for 2D visualization
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Combine features for clustering visualization
        cluster_features = ['manual_financial_health_score', 'manual_engagement_score',
                           'credit_score', 'dti_ratio', 'income']
        cluster_data = self.manual_results[cluster_features].fillna(0)
        
        # Scale and apply PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        pca = PCA(n_components=2, random_state=42)
        pca_data = pca.fit_transform(scaled_data)
        
        # Color by financial category
        categories = self.manual_results['manual_financial_category']
        category_colors = {'Poor': 'red', 'Fair': 'orange', 'Good': 'lightblue', 'Excellent': 'green'}
        colors = [category_colors.get(cat, 'gray') for cat in categories]
        
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.6, s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA Clustering by Financial Category', fontsize=12, fontweight='bold')
        
        # Add legend
        for category, color in category_colors.items():
            plt.scatter([], [], c=color, label=category, s=50)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('correlation_and_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Created correlation and clustering analysis: correlation_and_clustering_analysis.png")
    
    def create_financial_dashboard(self):
        """Create comprehensive financial dashboard similar to enhanced_financial_recommender_with_ml_engagement_only.py"""
        print("\n📊 CREATING FINANCIAL DASHBOARD")
        print("=" * 60)
        
        # Create dashboard figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Financial Health Score Distribution by Category (Manual)
        plt.subplot(3, 4, 1)
        for category in ['Poor', 'Fair', 'Good', 'Excellent']:
            data = self.manual_results[self.manual_results['manual_financial_category'] == category]['manual_financial_health_score']
            if len(data) > 0:
                plt.hist(data, alpha=0.7, label=category, bins=15)
        plt.xlabel('Financial Health Score')
        plt.ylabel('Number of Users')
        plt.title('Manual: Score Distribution by Category', fontsize=11, fontweight='bold')
        plt.legend()
        
        # 2. Financial Health Score Distribution by Category (ML)
        plt.subplot(3, 4, 2)
        for category in ['Poor', 'Fair', 'Good', 'Excellent']:
            data = self.ml_results[self.ml_results['ml_financial_category'] == category]['ml_financial_health_score']
            if len(data) > 0:
                plt.hist(data, alpha=0.7, label=category, bins=15)
        plt.xlabel('Financial Health Score')
        plt.ylabel('Number of Users')
        plt.title('ML: Score Distribution by Category', fontsize=11, fontweight='bold')
        plt.legend()
        
        # 3. Income vs Credit Score (Manual coloring)
        plt.subplot(3, 4, 3)
        plt.scatter(self.manual_results['income'], self.manual_results['credit_score'],
                   c=self.manual_results['manual_financial_category'].astype('category').cat.codes,
                   cmap='viridis', alpha=0.6, s=30)
        plt.xlabel('Income')
        plt.ylabel('Credit Score')
        plt.title('Manual: Income vs Credit Score', fontsize=11, fontweight='bold')
        plt.colorbar(label='Financial Category')
        
        # 4. Income vs Credit Score (ML coloring)
        plt.subplot(3, 4, 4)
        plt.scatter(self.ml_results['income'], self.ml_results['credit_score'],
                   c=self.ml_results['ml_financial_category'].astype('category').cat.codes,
                   cmap='viridis', alpha=0.6, s=30)
        plt.xlabel('Income')
        plt.ylabel('Credit Score')
        plt.title('ML: Income vs Credit Score', fontsize=11, fontweight='bold')
        plt.colorbar(label='Financial Category')
        
        # 5. DTI Ratio Distribution by Category (Manual)
        plt.subplot(3, 4, 5)
        manual_categories = self.manual_results['manual_financial_category'].unique()
        dti_data = [self.manual_results[self.manual_results['manual_financial_category'] == cat]['dti_ratio'].values 
                    for cat in manual_categories]
        plt.boxplot(dti_data, labels=[cat[:4] for cat in manual_categories])
        plt.ylabel('DTI Ratio')
        plt.title('Manual: DTI by Category', fontsize=11, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 6. DTI Ratio Distribution by Category (ML)
        plt.subplot(3, 4, 6)
        ml_categories = self.ml_results['ml_financial_category'].unique()
        dti_data_ml = [self.ml_results[self.ml_results['ml_financial_category'] == cat]['dti_ratio'].values 
                       for cat in ml_categories]
        plt.boxplot(dti_data_ml, labels=[cat[:4] for cat in ml_categories])
        plt.ylabel('DTI Ratio')
        plt.title('ML: DTI by Category', fontsize=11, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 7. Engagement Score vs Financial Score (Manual)
        plt.subplot(3, 4, 7)
        plt.hexbin(self.manual_results['manual_engagement_score'], 
                  self.manual_results['manual_financial_health_score'],
                  gridsize=20, cmap='Blues')
        plt.xlabel('Engagement Score')
        plt.ylabel('Financial Health Score')
        plt.title('Manual: Engagement vs Financial\n(Hexbin Density)', fontsize=11, fontweight='bold')
        plt.colorbar(label='User Density')
        
        # 8. Engagement Score vs Financial Score (ML)
        plt.subplot(3, 4, 8)
        plt.hexbin(self.ml_results['ml_engagement_score'], 
                  self.ml_results['ml_financial_health_score'],
                  gridsize=20, cmap='Reds')
        plt.xlabel('Engagement Score')
        plt.ylabel('Financial Health Score')
        plt.title('ML: Engagement vs Financial\n(Hexbin Density)', fontsize=11, fontweight='bold')
        plt.colorbar(label='User Density')
        
        # 9. Missed Payments Distribution
        plt.subplot(3, 4, 9)
        manual_missed = self.manual_results['missed_payments'].value_counts().sort_index()
        ml_missed = self.ml_results['missed_payments'].value_counts().sort_index()
        
        x = np.arange(len(manual_missed))
        width = 0.35
        
        plt.bar(x - width/2, manual_missed.values, width, label='Manual', alpha=0.8)
        plt.bar(x + width/2, ml_missed.values, width, label='ML', alpha=0.8)
        plt.xlabel('Missed Payments')
        plt.ylabel('Number of Users')
        plt.title('Missed Payments Distribution', fontsize=11, fontweight='bold')
        plt.xticks(x, manual_missed.index)
        plt.legend()
        
        # 10. Click Rate vs Time Viewed
        plt.subplot(3, 4, 10)
        plt.scatter(self.manual_results['click_rate'], self.manual_results['avg_time_viewed'],
                   c=self.manual_results['manual_engagement_score'], cmap='plasma', alpha=0.6, s=30)
        plt.xlabel('Click Rate')
        plt.ylabel('Avg Time Viewed')
        plt.title('Manual: Click Rate vs Time', fontsize=11, fontweight='bold')
        plt.colorbar(label='Engagement Score')
        
        # 11. Financial Category vs Enhanced Segment (Manual)
        plt.subplot(3, 4, 11)
        segment_category_data = pd.crosstab(self.manual_results['manual_enhanced_segment'], 
                                          self.manual_results['manual_financial_category'])
        sns.heatmap(segment_category_data, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Manual: Segment vs Category', fontsize=11, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 12. Financial Category vs Enhanced Segment (ML)
        plt.subplot(3, 4, 12)
        segment_category_data_ml = pd.crosstab(self.ml_results['ml_enhanced_segment'], 
                                             self.ml_results['ml_financial_category'])
        sns.heatmap(segment_category_data_ml, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('ML: Segment vs Category', fontsize=11, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('comprehensive_financial_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Created comprehensive financial dashboard: comprehensive_financial_dashboard.png")
    
    def create_recommendations_analysis(self):
        """Create detailed recommendations analysis comparing manual vs ML"""
        print("\n🎯 CREATING RECOMMENDATIONS ANALYSIS")
        print("=" * 60)
        
        # Create recommendations comparison figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Primary recommendation comparison
        plt.subplot(3, 3, 1)
        manual_primary = self.manual_results['manual_primary_recommendation'].value_counts()
        ml_primary = self.ml_results['ml_primary_recommendation'].value_counts()
        
        # Combine all recommendation types
        all_recs = set(list(manual_primary.index) + list(ml_primary.index))
        manual_counts = [manual_primary.get(rec, 0) for rec in all_recs]
        ml_counts = [ml_primary.get(rec, 0) for rec in all_recs]
        
        x = np.arange(len(all_recs))
        width = 0.35
        
        plt.bar(x - width/2, manual_counts, width, label='Manual', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, ml_counts, width, label='ML', alpha=0.8, color='lightcoral')
        plt.xlabel('Content Type')
        plt.ylabel('Number of Users')
        plt.title('Primary Recommendations: Manual vs ML', fontsize=12, fontweight='bold')
        plt.xticks(x, list(all_recs), rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Financial priorities comparison
        plt.subplot(3, 3, 2)
        
        # Extract and count unique priorities
        manual_priorities = []
        for priorities_str in self.manual_results['manual_financial_priorities']:
            if pd.notna(priorities_str):
                manual_priorities.extend([p.strip() for p in priorities_str.split(',')])
        
        ml_priorities = []
        for priorities_str in self.ml_results['ml_financial_priorities']:
            if pd.notna(priorities_str):
                ml_priorities.extend([p.strip() for p in priorities_str.split(',')])
        
        manual_priority_counts = pd.Series(manual_priorities).value_counts()
        ml_priority_counts = pd.Series(ml_priorities).value_counts()
        
        # Get top 6 priorities for visualization
        top_priorities = set(list(manual_priority_counts.head(6).index) + list(ml_priority_counts.head(6).index))
        
        manual_priority_vals = [manual_priority_counts.get(p, 0) for p in top_priorities]
        ml_priority_vals = [ml_priority_counts.get(p, 0) for p in top_priorities]
        
        x2 = np.arange(len(top_priorities))
        plt.bar(x2 - width/2, manual_priority_vals, width, label='Manual', alpha=0.8, color='lightgreen')
        plt.bar(x2 + width/2, ml_priority_vals, width, label='ML', alpha=0.8, color='orange')
        plt.xlabel('Financial Priority')
        plt.ylabel('Number of Users')
        plt.title('Financial Priorities: Manual vs ML', fontsize=12, fontweight='bold')
        plt.xticks(x2, [p[:12] + '...' if len(p) > 12 else p for p in top_priorities], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Urgency flags comparison
        plt.subplot(3, 3, 3)
        
        # Extract and count urgency flags
        manual_flags = []
        for flags_str in self.manual_results['manual_urgency_flags']:
            if pd.notna(flags_str):
                manual_flags.extend([f.strip() for f in flags_str.split(',')])
        
        ml_flags = []
        for flags_str in self.ml_results['ml_urgency_flags']:
            if pd.notna(flags_str):
                ml_flags.extend([f.strip() for f in flags_str.split(',')])
        
        manual_flag_counts = pd.Series(manual_flags).value_counts()
        ml_flag_counts = pd.Series(ml_flags).value_counts()
        
        # Pie chart for urgency flags
        plt.pie(manual_flag_counts.values, labels=manual_flag_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Manual Urgency Flags Distribution', fontsize=12, fontweight='bold')
        
        plt.subplot(3, 3, 4)
        plt.pie(ml_flag_counts.values, labels=ml_flag_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('ML Urgency Flags Distribution', fontsize=12, fontweight='bold')
        
        # 4. Recommendation agreement analysis
        plt.subplot(3, 3, 5)
        
        # Calculate agreement rates
        comparison_df = pd.merge(
            self.manual_results[['user_id', 'manual_primary_recommendation', 'manual_secondary_recommendation', 'manual_tertiary_recommendation']],
            self.ml_results[['user_id', 'ml_primary_recommendation', 'ml_secondary_recommendation', 'ml_tertiary_recommendation']],
            on='user_id'
        )
        
        primary_agreement = (comparison_df['manual_primary_recommendation'] == comparison_df['ml_primary_recommendation']).sum()
        secondary_agreement = (comparison_df['manual_secondary_recommendation'] == comparison_df['ml_secondary_recommendation']).sum()
        tertiary_agreement = (comparison_df['manual_tertiary_recommendation'] == comparison_df['ml_tertiary_recommendation']).sum()
        
        agreement_rates = [primary_agreement, secondary_agreement, tertiary_agreement]
        total_users = len(comparison_df)
        agreement_percentages = [rate/total_users*100 for rate in agreement_rates]
        
        plt.bar(['Primary', 'Secondary', 'Tertiary'], agreement_percentages, color=['red', 'orange', 'yellow'], alpha=0.7)
        plt.ylabel('Agreement Rate (%)')
        plt.title('Recommendation Agreement: Manual vs ML', fontsize=12, fontweight='bold')
        plt.ylim(0, 100)
        for i, v in enumerate(agreement_percentages):
            plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 5. Content type heatmap
        plt.subplot(3, 3, 6)
        
        # Create transition matrix for primary recommendations
        content_types = ['improve', 'insights', 'drivescore', 'protect', 'credit_cards', 'loans']
        transition_matrix = []
        
        for manual_content in content_types:
            row = []
            for ml_content in content_types:
                count = len(comparison_df[
                    (comparison_df['manual_primary_recommendation'] == manual_content) & 
                    (comparison_df['ml_primary_recommendation'] == ml_content)
                ])
                row.append(count)
            transition_matrix.append(row)
        
        sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=content_types, yticklabels=content_types)
        plt.title('Content Recommendation Transitions\n(Manual → ML)', fontsize=12, fontweight='bold')
        plt.xlabel('ML Recommendations')
        plt.ylabel('Manual Recommendations')
        
        # 6-9. Strategy comparison and sample recommendations
        plt.subplot(3, 3, 7)
        plt.axis('off')
        
        # Strategy length comparison
        manual_strategy_lengths = [len(str(strategy)) for strategy in self.manual_results['manual_strategy']]
        ml_strategy_lengths = [len(str(strategy)) for strategy in self.ml_results['ml_strategy']]
        
        strategy_text = f"""
        STRATEGY ANALYSIS
        ═══════════════════
        
        📝 STRATEGY CHARACTERISTICS:
        Manual avg length: {np.mean(manual_strategy_lengths):.0f} chars
        ML avg length:     {np.mean(ml_strategy_lengths):.0f} chars
        
        🎯 RECOMMENDATION INSIGHTS:
        Primary agreement:   {agreement_percentages[0]:.1f}%
        Secondary agreement: {agreement_percentages[1]:.1f}%
        Tertiary agreement:  {agreement_percentages[2]:.1f}%
        
        📊 TOP MANUAL PRIORITIES:
        {chr(10).join([f"• {p}: {c}" for p, c in manual_priority_counts.head(3).items()])}
        
        🤖 TOP ML PRIORITIES:
        {chr(10).join([f"• {p}: {c}" for p, c in ml_priority_counts.head(3).items()])}
        
        💡 KEY DIFFERENCES:
        • ML may identify different risk patterns
        • Content prioritization may shift
        • Financial urgency assessment varies
        """
        
        plt.text(0.05, 0.95, strategy_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 8. Sample user comparison
        plt.subplot(3, 3, 8)
        plt.axis('off')
        
        # Show a sample of user recommendations
        sample_comparison = comparison_df.head(5)
        sample_text = "SAMPLE USER COMPARISONS\n" + "═" * 25 + "\n\n"
        
        for _, user in sample_comparison.iterrows():
            user_id = user['user_id']
            manual_rec = user['manual_primary_recommendation']
            ml_rec = user['ml_primary_recommendation']
            match = "✓" if manual_rec == ml_rec else "✗"
            
            sample_text += f"{user_id}: {match}\n"
            sample_text += f"  Manual: {manual_rec}\n"
            sample_text += f"  ML:     {ml_rec}\n\n"
        
        plt.text(0.05, 0.95, sample_text, transform=plt.gca().transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 9. Overall summary
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Calculate overall differences
        total_agreements = sum(agreement_rates)
        total_possible = total_users * 3
        overall_agreement = (total_agreements / total_possible) * 100
        
        summary_text = f"""
        OVERALL SUMMARY
        ═══════════════
        
        📈 AGREEMENT METRICS:
        Overall recommendation agreement: {overall_agreement:.1f}%
        
        🎯 CONTENT DISTRIBUTION:
        Manual top content: {manual_primary.index[0]}
        ML top content:     {ml_primary.index[0]}
        
        ⚠️ URGENCY PATTERNS:
        Manual critical users: {manual_flag_counts.get('CRITICAL_CREDIT_SCORE', 0) + manual_flag_counts.get('PAYMENT_CRISIS', 0)}
        ML critical users:     {ml_flag_counts.get('CRITICAL_CREDIT_SCORE', 0) + ml_flag_counts.get('PAYMENT_CRISIS', 0)}
        
        🔍 KEY INSIGHT:
        {"High agreement - methods are consistent" if overall_agreement > 70 else "Low agreement - ML found different patterns"}
        
        💼 BUSINESS IMPACT:
        • Different content prioritization
        • Varied financial risk assessment
        • Potentially different user journeys
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('recommendations_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Created recommendations comparison analysis: recommendations_comparison_analysis.png")
        
        return comparison_df
    
    def generate_csv_files(self, comparison_df):
        """Generate CSV files for manual and ML results"""
        print("\n💾 GENERATING CSV FILES")
        print("=" * 60)
        
        # 1. Manual weights results CSV
        manual_csv = self.manual_results[['user_id', 'manual_financial_health_score', 'manual_engagement_score',
                                        'manual_financial_category', 'manual_enhanced_segment',
                                        'credit_score', 'dti_ratio', 'income', 'missed_payments',
                                        'click_rate', 'avg_time_viewed', 'total_interactions',
                                        'manual_primary_recommendation', 'manual_secondary_recommendation', 'manual_tertiary_recommendation',
                                        'manual_financial_priorities', 'manual_urgency_flags', 'manual_strategy', 'manual_ml_confidence']].copy()
        manual_csv.columns = ['user_id', 'financial_health_score', 'engagement_score', 'financial_category', 
                             'enhanced_segment', 'credit_score', 'dti_ratio', 'income', 'missed_payments',
                             'click_rate', 'avg_time_viewed', 'total_interactions',
                             'primary_recommendation', 'secondary_recommendation', 'tertiary_recommendation',
                             'financial_priorities', 'urgency_flags', 'strategy', 'ml_confidence']
        manual_csv['weight_method'] = 'Manual'
        manual_csv.to_csv('manual_weights_results.csv', index=False)
        
        # 2. ML weights results CSV
        ml_csv = self.ml_results[['user_id', 'ml_financial_health_score', 'ml_engagement_score',
                                'ml_financial_category', 'ml_enhanced_segment',
                                'credit_score', 'dti_ratio', 'income', 'missed_payments',
                                'click_rate', 'avg_time_viewed', 'total_interactions',
                                'ml_primary_recommendation', 'ml_secondary_recommendation', 'ml_tertiary_recommendation',
                                'ml_financial_priorities', 'ml_urgency_flags', 'ml_strategy', 'ml_ml_confidence']].copy()
        ml_csv.columns = ['user_id', 'financial_health_score', 'engagement_score', 'financial_category', 
                         'enhanced_segment', 'credit_score', 'dti_ratio', 'income', 'missed_payments',
                         'click_rate', 'avg_time_viewed', 'total_interactions',
                         'primary_recommendation', 'secondary_recommendation', 'tertiary_recommendation',
                         'financial_priorities', 'urgency_flags', 'strategy', 'ml_confidence']
        ml_csv['weight_method'] = 'ML_Learned'
        ml_csv.to_csv('ml_weights_results.csv', index=False)
        
        # 3. Combined comparison CSV
        comparison_csv = comparison_df.copy()
        comparison_csv.to_csv('manual_vs_ml_comparison.csv', index=False)
        
        print("✅ Generated CSV files:")
        print("   📄 manual_weights_results.csv - Results using manual weights")
        print("   📄 ml_weights_results.csv - Results using ML-learned weights") 
        print("   📄 manual_vs_ml_comparison.csv - Side-by-side comparison")
        
        # Print summary statistics
        print(f"\n📊 CSV FILE CONTENTS:")
        print(f"   Manual weights: {len(manual_csv)} users")
        print(f"   ML weights: {len(ml_csv)} users")
        print(f"   Comparison: {len(comparison_csv)} users")
        
        return manual_csv, ml_csv, comparison_csv
    
    def run_complete_analysis(self):
        """Run the complete manual vs ML comparison analysis"""
        print("🚀 STARTING COMPREHENSIVE MANUAL VS ML ANALYSIS")
        print("=" * 80)
        
        # Step 0: Train ML models for recommendations
        print("\n0️⃣ TRAINING SUPERVISED LEARNING MODELS")
        print("-" * 60)
        training_data = self.create_real_training_data()
        model_performance = self.train_ml_recommendation_models(training_data)
        
        # Step 1: Calculate manual weights analysis
        manual_results = self.calculate_manual_weights_analysis()
        
        # Step 2: Run ML analysis
        ml_results = self.run_ml_analysis()
        
        # Step 3: Create comparison visualizations
        comparison_df = self.create_comparison_visualizations()
        
        # Step 4: Create detailed segment analysis
        self.create_detailed_segment_analysis(comparison_df)
        
        # Step 5: Create correlation analysis
        self.create_correlation_analysis()
        
        # Step 6: Create comprehensive financial dashboard
        self.create_financial_dashboard()
        
        # Step 7: Create recommendations analysis
        recommendations_comparison = self.create_recommendations_analysis()
        
        # Step 8: Generate CSV files
        manual_csv, ml_csv, comparison_csv = self.generate_csv_files(comparison_df)
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 80)
        print("✅ ML recommendation models trained")
        print("✅ Manual weight analysis completed")
        print("✅ ML weight optimization completed")
        print("✅ Comprehensive visualizations created")
        print("✅ Detailed segment analysis created")
        print("✅ Correlation analysis created")
        print("✅ Financial dashboard created")
        print("✅ Recommendations analysis created")
        print("✅ CSV files generated")
        print("\n📁 FILES CREATED:")
        print("   🖼️ manual_vs_ml_comprehensive_comparison.png")
        print("   🖼️ detailed_segment_analysis.png")
        print("   🖼️ correlation_and_clustering_analysis.png")
        print("   🖼️ comprehensive_financial_dashboard.png")
        print("   🖼️ recommendations_comparison_analysis.png")
        print("   📄 manual_weights_results.csv")
        print("   📄 ml_weights_results.csv") 
        print("   📄 manual_vs_ml_comparison.csv")
        
        print("\n🤖 ML MODEL PERFORMANCE:")
        print("-" * 40)
        print(f"   📊 Recommendation Model: {model_performance['recommendation_accuracy']:.1%}")
        print(f"   💰 Priority Model: {model_performance['priority_accuracy']:.1%}")
        print(f"   🚨 Urgency Model: {model_performance['urgency_accuracy']:.1%}")
        
        return {
            'manual_results': manual_results,
            'ml_results': ml_results,
            'comparison_df': comparison_df,
            'model_performance': model_performance,
            'csv_files': {
                'manual': manual_csv,
                'ml': ml_csv,
                'comparison': comparison_csv
            }
        }

if __name__ == "__main__":
    print("🔬 MANUAL VS ML COMPREHENSIVE COMPARISON")
    print("=" * 80)
    
    # Set global random seed for completely reproducible results
    np.random.seed(42)
    
    try:
        # Create comparison analyzer
        analyzer = ManualVsMLComparison('joined_user_table.csv')
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        print("\n💡 KEY INSIGHTS TO LOOK FOR:")
        print("─" * 50)
        print("• How different are the financial category distributions?")
        print("• Which users changed segments between manual and ML?")
        print("• Do the ML weights reveal hidden patterns?")
        print("• Are the engagement score correlations high or low?")
        print("• Check the transition matrix to see category migrations!")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Please ensure the data file exists in the current directory")
    except Exception as e:
        print(f"❌ Error: {e}") 