"""
ML-Enhanced Financial Recommender with Hybrid Approach
- ML model for content preference prediction based on user attributes
- Rules-based financial priority and urgency flag detection
- Hybrid scoring system combining ML predictions with financial context
- Model performance evaluation and comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import from the refactored version
from refactored_financial_recommender import (
    FinancialConfig, EngagementConfig, ClusteringConfig,
    UserFinancialProfile, UserEngagementProfile,
    FinancialHealthCalculator, EngagementCalculator,
    PriorityBasedSegmentation, VisualizationFactory,
    DTIAnalyzer
)

# ================================
# ML MODEL CONFIGURATION
# ================================

@dataclass
class MLConfig:
    """Configuration for ML models"""
    # Model selection
    content_model_type: str = "random_forest"  # random_forest, logistic_regression, decision_tree
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Feature engineering
    min_content_interactions: int = 3  # Minimum interactions to include content type
    feature_selection_threshold: float = 0.01  # Importance threshold for feature selection
    
    # Hybrid scoring weights
    ml_prediction_weight: float = 0.6  # Weight for ML content predictions
    rules_adjustment_weight: float = 0.4  # Weight for rules-based financial adjustments

# ================================
# ML CONTENT PREDICTION ENGINE
# ================================

class MLContentPredictor:
    """Machine Learning engine for predicting content preferences from user attributes"""
    
    def __init__(self, ml_config: MLConfig):
        self.ml_config = ml_config
        self.models = {}
        self.label_encoders = {}
        self.feature_scaler = StandardScaler()
        self.content_types = ['improve', 'insights', 'drivescore', 'protect', 'credit_cards', 'loans']
        self.feature_columns = []
        self.model_performance = {}
        
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data from engagement history"""
        print("🔬 Preparing ML training data...")
        
        # Create user-content interaction matrix
        user_content_matrix = df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
        
        # Convert to binary preferences (1 if user interacted with content type, 0 otherwise)
        # But only for content types with sufficient interactions
        content_preferences = pd.DataFrame(index=user_content_matrix.index)
        
        for content_type in self.content_types:
            if content_type in user_content_matrix.columns:
                # Binary: 1 if user has >= min_interactions with this content type
                content_preferences[f'prefers_{content_type}'] = (
                    user_content_matrix[content_type] >= self.ml_config.min_content_interactions
                ).astype(int)
            else:
                content_preferences[f'prefers_{content_type}'] = 0
        
        # Get user attributes (features)
        user_attributes = df.groupby('user_id').first()[
            ['credit_score', 'dti_ratio', 'income', 'total_debt', 'missed_payments', 
             'has_ccj', 'has_mortgage', 'has_car']
        ]
        
        # Add engagement features
        engagement_features = df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        })
        engagement_features.columns = ['avg_time', 'total_time', 'click_rate', 'total_clicks', 'unique_content']
        
        # Combine features
        features = pd.concat([user_attributes, engagement_features], axis=1).fillna(0)
        
        # Align indices
        common_users = features.index.intersection(content_preferences.index)
        features = features.loc[common_users]
        content_preferences = content_preferences.loc[common_users]
        
        print(f"✅ Training data prepared: {len(features)} users, {len(features.columns)} features")
        print(f"Content preference distribution:")
        for col in content_preferences.columns:
            pref_count = content_preferences[col].sum()
            pref_pct = (pref_count / len(content_preferences)) * 100
            print(f"  {col}: {pref_count} users ({pref_pct:.1f}%)")
        
        return features, content_preferences
    
    def train_content_models(self, features: pd.DataFrame, content_preferences: pd.DataFrame):
        """Train ML models for each content type"""
        print(f"\n🤖 Training {self.ml_config.content_model_type} models for content prediction...")
        
        # Store feature columns for later use
        self.feature_columns = features.columns.tolist()
        
        # Encode categorical features
        features_encoded = features.copy()
        categorical_cols = ['has_ccj', 'has_mortgage', 'has_car']
        
        for col in categorical_cols:
            if col in features_encoded.columns:
                le = LabelEncoder()
                features_encoded[col] = le.fit_transform(features_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features_encoded)
        features_scaled_df = pd.DataFrame(features_scaled, columns=features_encoded.columns, index=features_encoded.index)
        
        # Train model for each content type
        for content_type in self.content_types:
            target_col = f'prefers_{content_type}'
            if target_col not in content_preferences.columns:
                continue
                
            print(f"  Training model for {content_type}...")
            
            y = content_preferences[target_col]
            
            # Skip if not enough positive examples
            if y.sum() < 5:
                print(f"    ⚠️ Skipping {content_type} - insufficient positive examples ({y.sum()})")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled_df, y, test_size=self.ml_config.test_size, 
                random_state=self.ml_config.random_state, stratify=y if y.sum() > 1 else None
            )
            
            # Select model
            if self.ml_config.content_model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100, random_state=self.ml_config.random_state,
                    class_weight='balanced'
                )
            elif self.ml_config.content_model_type == "logistic_regression":
                model = LogisticRegression(
                    random_state=self.ml_config.random_state, class_weight='balanced',
                    max_iter=1000
                )
            else:  # decision_tree
                model = DecisionTreeClassifier(
                    random_state=self.ml_config.random_state, class_weight='balanced'
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, features_scaled_df, y, cv=self.ml_config.cv_folds)
            
            # Store model and performance
            self.models[content_type] = model
            self.model_performance[content_type] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'positive_examples': y.sum(),
                'total_examples': len(y)
            }
            
            print(f"    ✅ {content_type}: Accuracy={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        print(f"\n✅ ML model training complete! Trained {len(self.models)} content models")
    
    def predict_content_preferences(self, user_attributes: pd.DataFrame) -> pd.DataFrame:
        """Predict content preferences for users using trained ML models"""
        if not self.models:
            raise ValueError("Models not trained yet. Call train_content_models() first.")
        
        # Prepare features
        features_encoded = user_attributes.copy()
        
        # Encode categorical features
        categorical_cols = ['has_ccj', 'has_mortgage', 'has_car']
        for col in categorical_cols:
            if col in features_encoded.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                features_encoded[col] = le.transform(features_encoded[col].astype(str))
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in features_encoded.columns:
                features_encoded[col] = 0
        
        # Reorder columns to match training
        features_encoded = features_encoded[self.feature_columns]
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features_encoded)
        
        # Predict preferences
        predictions = pd.DataFrame(index=user_attributes.index)
        
        for content_type, model in self.models.items():
            # Get probability of preferring this content type
            pred_proba = model.predict_proba(features_scaled)
            # Take probability of class 1 (prefers content)
            if pred_proba.shape[1] > 1:
                predictions[f'ml_score_{content_type}'] = pred_proba[:, 1]
            else:
                # Handle case where only one class exists
                predictions[f'ml_score_{content_type}'] = 0.5
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance for each content model"""
        importance_data = {}
        
        for content_type, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                importance_data[content_type] = importance_df
        
        return importance_data
    
    def print_model_summary(self):
        """Print summary of model performance"""
        print("\n🎯 ML MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for content_type, perf in self.model_performance.items():
            print(f"\n{content_type.upper()}:")
            print(f"  Accuracy: {perf['accuracy']:.3f}")
            print(f"  Cross-validation: {perf['cv_mean']:.3f} ± {perf['cv_std']:.3f}")
            print(f"  Training examples: {perf['positive_examples']}/{perf['total_examples']} positive")
        
        print(f"\nOverall model quality:")
        accuracies = [perf['accuracy'] for perf in self.model_performance.values()]
        cv_scores = [perf['cv_mean'] for perf in self.model_performance.values()]
        print(f"  Mean accuracy: {np.mean(accuracies):.3f}")
        print(f"  Mean CV score: {np.mean(cv_scores):.3f}")

# ================================
# HYBRID RECOMMENDATION ENGINE
# ================================

class HybridRecommendationEngine:
    """Combines ML content predictions with rules-based financial prioritization"""
    
    def __init__(self, financial_config: FinancialConfig, ml_config: MLConfig):
        self.financial_config = financial_config
        self.ml_config = ml_config
        self.ml_predictor = MLContentPredictor(ml_config)
        self.content_types = ['improve', 'insights', 'drivescore', 'protect', 'credit_cards', 'loans']
    
    def train_ml_models(self, df: pd.DataFrame):
        """Train the ML content prediction models"""
        features, content_preferences = self.ml_predictor.prepare_training_data(df)
        self.ml_predictor.train_content_models(features, content_preferences)
    
    def generate_hybrid_recommendations(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations using hybrid ML + rules approach"""
        print("🔄 Generating hybrid ML + rules-based recommendations...")
        
        # Step 1: Get ML content predictions
        ml_predictions = self.ml_predictor.predict_content_preferences(user_features)
        
        recommendations = []
        
        for user_id, user_data in user_features.iterrows():
            segment = user_data['enhanced_segment']
            financial_cat = user_data['financial_category']
            credit_score = user_data['credit_score']
            dti_ratio = user_data['dti_ratio']
            has_ccj = user_data['has_ccj']
            missed_payments = user_data['missed_payments']
            
            # Step 2: Rules-based financial priorities (unchanged)
            financial_priorities = self.get_financial_priorities(user_data)
            urgency_flags = self.get_urgency_flags(user_data)
            
            # Step 3: Hybrid scoring - combine ML predictions with financial rules
            content_scores = {}
            
            for content_type in self.content_types:
                # Get ML prediction score
                ml_score = ml_predictions.loc[user_id, f'ml_score_{content_type}'] if f'ml_score_{content_type}' in ml_predictions.columns else 0.5
                
                # Apply rules-based financial adjustments
                financial_adjustment = self._get_financial_adjustment(content_type, user_data)
                
                # Hybrid scoring: weighted combination
                hybrid_score = (
                    self.ml_config.ml_prediction_weight * ml_score +
                    self.ml_config.rules_adjustment_weight * financial_adjustment
                )
                
                content_scores[content_type] = hybrid_score
            
            # Sort by hybrid scores
            sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Generate strategy
            strategy = self.get_strategy_by_segment(segment, user_data)
            
            recommendations.append({
                'user_id': user_id,
                'enhanced_segment': segment,
                'financial_category': financial_cat,
                'primary_recommendation': sorted_content[0][0],
                'secondary_recommendation': sorted_content[1][0],
                'tertiary_recommendation': sorted_content[2][0],
                'primary_ml_score': ml_predictions.loc[user_id, f'ml_score_{sorted_content[0][0]}'] if f'ml_score_{sorted_content[0][0]}' in ml_predictions.columns else 0.5,
                'primary_hybrid_score': sorted_content[0][1],
                'financial_priorities': ', '.join(financial_priorities),
                'urgency_flags': ', '.join(urgency_flags),
                'strategy': strategy,
                'credit_score': credit_score,
                'dti_ratio': dti_ratio,
                'financial_health_score': user_data['financial_health_score'],
                'engagement_score': user_data['engagement_score'],
                'recommendation_method': 'ML_Hybrid'
            })
        
        return pd.DataFrame(recommendations)
    
    def _get_financial_adjustment(self, content_type: str, user_data: pd.Series) -> float:
        """Get financial context adjustment for content type (0-1 scale)"""
        base_score = 0.5  # Neutral baseline
        missed_payments = user_data['missed_payments']
        dti_ratio = user_data['dti_ratio']
        credit_score = user_data['credit_score']
        financial_cat = user_data['financial_category']
        has_ccj = user_data['has_ccj']
        
        # Apply the same financial logic as the rules-based system
        # PRIORITY 1: FINN_DIFF (Missed Payments >= 2)
        if missed_payments >= self.financial_config.finn_diff_threshold:
            if content_type == 'improve':
                return 0.95  # Very high priority
            elif content_type == 'insights':
                return 0.85
            elif content_type == 'credit_cards':
                return 0.15  # Discourage
            elif content_type == 'loans':
                return 0.25  # Discourage
            elif content_type == 'protect':
                return 0.65
            else:
                return 0.5
        
        # PRIORITY 2: HIGH DTI (>= 50%)
        elif dti_ratio >= self.financial_config.high_dti_threshold:
            if content_type == 'improve':
                return 0.98  # Highest priority
            elif content_type == 'insights':
                return 0.90
            elif content_type == 'loans':
                return 0.10  # Strongly discourage
            elif content_type == 'credit_cards':
                return 0.05  # Strongly discourage
            elif content_type == 'protect':
                return 0.20
            else:
                return 0.4
        
        # Other financial context adjustments
        adjustments = {
            'improve': 0.8 if credit_score < 650 else 0.5,
            'protect': 0.75 if financial_cat == "Excellent" else 0.5,
            'loans': 0.25 if dti_ratio > 0.6 else 0.5,
            'credit_cards': 0.15 if has_ccj else 0.5,
            'drivescore': 0.8 if financial_cat == "Poor" else 0.5,
            'insights': 0.8 if missed_payments > 2 else 0.5
        }
        
        return adjustments.get(content_type, 0.5)
    
    def get_financial_priorities(self, user_data: pd.Series) -> List[str]:
        """Identify financial priorities for a user (unchanged from rules-based)"""
        priorities = []
        
        if user_data['missed_payments'] >= self.financial_config.finn_diff_threshold:
            priorities.append("URGENT_PAYMENT_MANAGEMENT")
        elif user_data['dti_ratio'] >= self.financial_config.high_dti_threshold:
            priorities.append("URGENT_DTI_REDUCTION")
        
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
    
    def get_urgency_flags(self, user_data: pd.Series) -> List[str]:
        """Identify urgent financial issues (unchanged from rules-based)"""
        flags = []
        
        if user_data['dti_ratio'] > 0.8:
            flags.append("HIGH_DEBT_BURDEN")
        if user_data['credit_score'] < 500:
            flags.append("CRITICAL_CREDIT_SCORE")
        if user_data['missed_payments'] > 4:
            flags.append("PAYMENT_CRISIS")
        if user_data['has_ccj']:
            flags.append("LEGAL_ACTION")
        
        return flags if flags else ["STABLE_FINANCIAL_POSITION"]
    
    def get_strategy_by_segment(self, segment: str, user_data: pd.Series) -> str:
        """Get tailored strategy based on enhanced segment (unchanged from rules-based)"""
        strategies = {
            "Debt_Management_Priority": f"🚨 CRITICAL: DTI {user_data['dti_ratio']:.1%} - URGENT debt reduction required. ML model recommends content with financial override priority.",
            
            "Payment_Recovery_Priority": f"🚨 PAYMENT ISSUES: {user_data['missed_payments']} missed payments detected - URGENT payment management required. ML predictions adjusted for payment crisis.",
            
            "Premium_Engaged": f"ML-driven premium content strategy. Machine learning identified content preferences combined with wealth management focus. Credit score: {user_data['credit_score']}.",
            
            "Growth_Focused": f"Hybrid ML + growth strategy. ML predictions weighted with moderate financial complexity content. Current DTI: {user_data['dti_ratio']:.2f}.",
            
            "Recovery_Engaged": f"ML-enhanced recovery content with high engagement prediction. Combines user preference modeling with debt management priority. Target: Credit improvement from {user_data['credit_score']}.",
            
            "Premium_Moderate": f"ML premium content recommendations with clear value propositions. Algorithmic content selection balanced with wealth building advice.",
            
            "Mainstream": f"ML-balanced content strategy for users with decent financial health. Machine learning predictions guide practical financial advice selection.",
            
            "Recovery_Moderate": f"ML-accessible financial recovery content. Algorithm simplifies content complexity while maintaining financial priority adjustments.",
            
            "Financial_Priority": f"Crisis-override mode: Financial rules take precedence over ML predictions. Focus on critical issues with ML-guided secondary content.",
            
            "Activation_Needed": f"ML-driven engagement building with basic financial education. Algorithm identifies content preferences while building financial awareness."
        }
        
        return strategies.get(segment, "Hybrid ML + rules-based guidance optimized for user profile.")

# ================================
# ML-ENHANCED MAIN RECOMMENDER
# ================================

class MLEnhancedFinancialRecommender:
    """ML-Enhanced Financial Recommender with hybrid content prediction"""
    
    def __init__(self, 
                 csv_file: str,
                 financial_config: Optional[FinancialConfig] = None,
                 engagement_config: Optional[EngagementConfig] = None,
                 clustering_config: Optional[ClusteringConfig] = None,
                 ml_config: Optional[MLConfig] = None):
        
        self.df = pd.read_csv(csv_file)
        self.financial_config = financial_config or FinancialConfig()
        self.engagement_config = engagement_config or EngagementConfig()
        self.clustering_config = clustering_config or ClusteringConfig()
        self.ml_config = ml_config or MLConfig()
        
        # Components
        self.financial_calculator = FinancialHealthCalculator(self.financial_config)
        self.engagement_calculator = EngagementCalculator(self.engagement_config)
        self.segmentation_strategy = PriorityBasedSegmentation(self.financial_config, self.engagement_config)
        self.hybrid_engine = HybridRecommendationEngine(self.financial_config, self.ml_config)
        
        # Data
        self.user_features = None
        self.scaler = StandardScaler()
        
    def create_user_features(self) -> pd.DataFrame:
        """Create enhanced user features (reuse from refactored version)"""
        print("Creating enhanced user features for ML training...")
        
        # Engagement features
        engagement_features = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).round(2)
        
        engagement_features.columns = [
            'avg_time_viewed', 'total_time_viewed', 'total_interactions',
            'click_rate', 'total_clicks', 'unique_content_viewed'
        ]
        
        # Financial features
        user_financial = self.df.groupby('user_id').first()[
            ['total_debt', 'credit_score', 'missed_payments', 
             'has_mortgage', 'has_car', 'has_ccj', 'dti_ratio', 'income']
        ]
        
        # Calculate financial health scores
        financial_scores = []
        financial_categories = []
        
        for user_id, user_data in user_financial.iterrows():
            profile = UserFinancialProfile(
                user_id=user_id,
                credit_score=user_data['credit_score'],
                dti_ratio=user_data['dti_ratio'],
                income=user_data['income'],
                total_debt=user_data['total_debt'],
                missed_payments=user_data['missed_payments'],
                has_ccj=user_data['has_ccj'],
                has_mortgage=user_data['has_mortgage'],
                has_car=user_data['has_car']
            )
            
            health_score = self.financial_calculator.calculate_health_score(profile)
            category = self.financial_calculator.categorize_health(health_score)
            
            financial_scores.append(health_score)
            financial_categories.append(category)
        
        user_financial['financial_health_score'] = financial_scores
        user_financial['financial_category'] = financial_categories
        
        # Content preferences (for ML training)
        content_preferences = self.df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
        content_preferences = content_preferences.div(content_preferences.sum(axis=1), axis=0)
        content_preferences.columns = [f'pref_{col}' for col in content_preferences.columns]
        
        # Combine features
        self.user_features = pd.concat([
            engagement_features,
            content_preferences, 
            user_financial
        ], axis=1).fillna(0)
        
        return self.user_features
    
    def perform_segmentation(self) -> pd.DataFrame:
        """Perform user segmentation (reuse from refactored version)"""
        print("Performing enhanced segmentation...")
        
        if self.user_features is None:
            self.create_user_features()
        
        engagement_scores = []
        segments = []
        
        for user_id, user_data in self.user_features.iterrows():
            eng_profile = UserEngagementProfile(
                user_id=user_id,
                click_rate=user_data['click_rate'],
                avg_time_viewed=user_data['avg_time_viewed'],
                total_interactions=user_data['total_interactions'],
                unique_content_viewed=user_data['unique_content_viewed'],
                content_preferences={}
            )
            
            fin_profile = UserFinancialProfile(
                user_id=user_id,
                credit_score=user_data['credit_score'],
                dti_ratio=user_data['dti_ratio'],
                income=user_data['income'],
                total_debt=user_data['total_debt'],
                missed_payments=user_data['missed_payments'],
                has_ccj=user_data['has_ccj'],
                has_mortgage=user_data['has_mortgage'],
                has_car=user_data['has_car']
            )
            
            engagement_score = self.engagement_calculator.calculate_engagement_score(eng_profile)
            segment = self.segmentation_strategy.assign_segment(engagement_score, fin_profile)
            
            engagement_scores.append(engagement_score)
            segments.append(segment)
        
        self.user_features['engagement_score'] = engagement_scores
        self.user_features['enhanced_segment'] = segments
        
        return self.user_features
    
    def train_and_generate_ml_recommendations(self) -> pd.DataFrame:
        """Train ML models and generate hybrid recommendations"""
        print("🤖 Training ML models and generating hybrid recommendations...")
        
        # Ensure features and segmentation are ready
        if self.user_features is None or 'enhanced_segment' not in self.user_features.columns:
            self.perform_segmentation()
        
        # Train ML models
        self.hybrid_engine.train_ml_models(self.df)
        
        # Generate hybrid recommendations
        recommendations = self.hybrid_engine.generate_hybrid_recommendations(self.user_features)
        
        return recommendations
    
    def run_ml_enhanced_analysis(self, config_name: str = "ml_enhanced") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Run complete ML-enhanced analysis"""
        print(f"🚀 STARTING ML-ENHANCED ANALYSIS ({config_name.upper()})")
        print("=" * 70)
        
        # Create features and segmentation
        features = self.create_user_features()
        segmented_features = self.perform_segmentation()
        
        # Train ML and generate recommendations
        recommendations = self.train_and_generate_ml_recommendations()
        
        # Print ML model performance
        self.hybrid_engine.ml_predictor.print_model_summary()
        
        # Save results
        recommendations_filename = f"{config_name}_recommendations.csv"
        recommendations.to_csv(recommendations_filename, index=False)
        print(f"✅ ML recommendations saved to '{recommendations_filename}'")
        
        # Analyze segments
        analysis = segmented_features.groupby('enhanced_segment').agg({
            'financial_health_score': ['mean', 'std'],
            'credit_score': ['mean', 'min', 'max'],
            'dti_ratio': ['mean', 'std'],
            'income': ['mean', 'median'],
            'engagement_score': ['mean']
        }).round(2)
        
        # Create visualizations
        print(f"\n📊 CREATING {config_name.upper()} VISUALIZATIONS...")
        financial_dashboard_fig = VisualizationFactory.create_financial_dashboard(segmented_features)
        correlation_heatmap_fig = VisualizationFactory.create_correlation_heatmap(segmented_features)
        clustering_visualization_fig = VisualizationFactory.create_clustering_visualization(segmented_features, self.clustering_config)
        
        VisualizationFactory.display_and_save_figure(
            financial_dashboard_fig, f"{config_name}_financial_dashboard.png"
        )
        VisualizationFactory.display_and_save_figure(
            correlation_heatmap_fig, f"{config_name}_correlation_heatmap.png"
        )
        VisualizationFactory.display_and_save_figure(
            clustering_visualization_fig, f"{config_name}_clustering_visualization.png"
        )
        
        # Print sample recommendations
        print(f"\n🎯 SAMPLE ML-ENHANCED RECOMMENDATIONS:")
        self._print_ml_recommendations(recommendations, n_samples=5)
        
        print(f"\n✅ {config_name.upper()} ML-ENHANCED ANALYSIS COMPLETE!")
        return segmented_features, recommendations, analysis
    
    def _print_ml_recommendations(self, recommendations_df: pd.DataFrame, n_samples: int = 10):
        """Print sample ML-enhanced recommendations"""
        print("\n" + "=" * 80)
        print("ML-ENHANCED FINANCIALLY-AWARE RECOMMENDATIONS")
        print("=" * 80)
        
        for _, user in recommendations_df.head(n_samples).iterrows():
            print(f"\nUser: {user['user_id']}")
            print(f"Enhanced Segment: {user['enhanced_segment']}")
            print(f"Financial Category: {user['financial_category']}")
            print(f"Credit Score: {user['credit_score']} | DTI: {user['dti_ratio']:.2f} | Health Score: {user['financial_health_score']:.2f}")
            print(f"Primary Rec: {user['primary_recommendation']} (ML Score: {user['primary_ml_score']:.3f}, Hybrid: {user['primary_hybrid_score']:.3f})")
            print(f"Method: {user['recommendation_method']}")
            print(f"Financial Priorities: {user['financial_priorities']}")
            print(f"Urgency Flags: {user['urgency_flags']}")
            print(f"Strategy: {user['strategy'][:120]}...")
            print("-" * 80)
    
    def compare_with_rules_based(self) -> pd.DataFrame:
        """Compare ML-enhanced recommendations with pure rules-based approach"""
        print("\n🔄 Comparing ML-enhanced vs rules-based recommendations...")
        
        # Import rules-based engine from refactored version
        from refactored_financial_recommender import RecommendationEngine
        rules_engine = RecommendationEngine(self.financial_config)
        
        # Generate both types of recommendations
        ml_recommendations = self.train_and_generate_ml_recommendations()
        rules_recommendations = rules_engine.generate_financial_content_recommendations(self.user_features)
        
        # Compare primary recommendations
        comparison = pd.DataFrame({
            'user_id': ml_recommendations['user_id'],
            'ml_primary': ml_recommendations['primary_recommendation'],
            'rules_primary': rules_recommendations['primary_recommendation'],
            'enhanced_segment': ml_recommendations['enhanced_segment'],
            'financial_category': ml_recommendations['financial_category'],
            'ml_hybrid_score': ml_recommendations['primary_hybrid_score'],
            'ml_pure_score': ml_recommendations['primary_ml_score']
        })
        
        # Calculate agreement
        agreement = (comparison['ml_primary'] == comparison['rules_primary']).mean()
        print(f"📊 Primary recommendation agreement: {agreement:.1%}")
        
        # Agreement by segment
        segment_agreement = comparison.groupby('enhanced_segment').apply(
            lambda x: (x['ml_primary'] == x['rules_primary']).mean()
        ).round(3)
        
        print("\n📈 Agreement by segment:")
        for segment, agree_rate in segment_agreement.items():
            print(f"  {segment}: {agree_rate:.1%}")
        
        # Save comparison
        comparison.to_csv('ml_vs_rules_comparison.csv', index=False)
        print("✅ Comparison saved to 'ml_vs_rules_comparison.csv'")
        
        return comparison

if __name__ == "__main__":
    print("🤖 ML-ENHANCED FINANCIAL RECOMMENDER")
    print("=" * 60)
    
    try:
        # Test different ML configurations
        print("\n🔬 TESTING DIFFERENT ML CONFIGURATIONS")
        
        # Configuration 1: Random Forest with balanced weighting
        print("\n📊 CONFIGURATION 1: Random Forest (Balanced)")
        ml_config_rf = MLConfig(
            content_model_type="random_forest",
            ml_prediction_weight=0.6,
            rules_adjustment_weight=0.4
        )
        
        ml_recommender_rf = MLEnhancedFinancialRecommender(
            'joined_user_table.csv',
            ml_config=ml_config_rf
        )
        features_rf, recommendations_rf, analysis_rf = ml_recommender_rf.run_ml_enhanced_analysis("ml_random_forest")
        
        # Configuration 2: Logistic Regression with ML emphasis
        print("\n📊 CONFIGURATION 2: Logistic Regression (ML Emphasis)")
        ml_config_lr = MLConfig(
            content_model_type="logistic_regression",
            ml_prediction_weight=0.8,
            rules_adjustment_weight=0.2
        )
        
        ml_recommender_lr = MLEnhancedFinancialRecommender(
            'joined_user_table.csv',
            ml_config=ml_config_lr
        )
        features_lr, recommendations_lr, analysis_lr = ml_recommender_lr.run_ml_enhanced_analysis("ml_logistic_regression")
        
        # Configuration 3: Decision Tree with rules emphasis
        print("\n📊 CONFIGURATION 3: Decision Tree (Rules Emphasis)")
        ml_config_dt = MLConfig(
            content_model_type="decision_tree",
            ml_prediction_weight=0.3,
            rules_adjustment_weight=0.7
        )
        
        ml_recommender_dt = MLEnhancedFinancialRecommender(
            'joined_user_table.csv',
            ml_config=ml_config_dt
        )
        features_dt, recommendations_dt, analysis_dt = ml_recommender_dt.run_ml_enhanced_analysis("ml_decision_tree")
        
        # Compare ML vs Rules-based
        print("\n🔄 COMPARING ML-ENHANCED VS PURE RULES-BASED")
        comparison = ml_recommender_rf.compare_with_rules_based()
        
        print("\n🎉 ML-ENHANCED ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\n📁 Generated files:")
        print("  • ml_random_forest_recommendations.csv")
        print("  • ml_logistic_regression_recommendations.csv") 
        print("  • ml_decision_tree_recommendations.csv")
        print("  • ml_vs_rules_comparison.csv")
        print("  • Multiple dashboard and visualization files")
        
        print("\n💡 KEY INSIGHTS:")
        print("✅ ML models predict content preferences from user attributes")
        print("✅ Rules-based financial priorities override ML when needed")
        print("✅ Hybrid scoring balances ML predictions with financial context")
        print("✅ Different ML algorithms can be compared for effectiveness")
        print("✅ Agreement analysis shows where ML and rules align/differ")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Run refactored_financial_recommender.py first to generate the data")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 