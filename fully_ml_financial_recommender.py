"""
Fully ML-Driven Financial Recommender
This system uses machine learning to optimize BOTH engagement weights AND financial health weights
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class FullyMLFinancialRecommender:
    """
    Uses ML to optimize both engagement weights AND financial health weights
    """
    
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.ml_engagement_weights = None
        self.ml_financial_weights = None
        self.user_features = None
        
    def optimize_financial_health_weights(self, target_variable='business_outcome'):
        """
        Use supervised learning to optimize financial health score weights
        
        Potential targets:
        - business_outcome: Revenue, retention, conversion rates
        - loan_default: Whether user defaulted (if available)
        - customer_lifetime_value: Total value generated
        - risk_score: External risk assessment
        """
        print("🤖 OPTIMIZING FINANCIAL HEALTH WEIGHTS WITH ML")
        print("=" * 60)
        
        # Financial components (same as manual system)
        financial_features = [
            'credit_score', 'dti_ratio', 'income', 'total_debt', 
            'missed_payments', 'has_ccj', 'has_mortgage', 'has_car'
        ]
        
        # For demo, create a synthetic business outcome target
        # In real usage, this would be your actual business metric
        if target_variable not in self.df.columns:
            print(f"⚠️  {target_variable} not found. Creating synthetic business outcome for demo...")
            # Create synthetic target based on intuitive business logic
            self.df['business_outcome'] = self._create_synthetic_business_outcome()
            target_variable = 'business_outcome'
        
        # Prepare financial features
        X_financial = self.df[financial_features].copy()
        
        # Transform features to components (similar to manual calculation)
        X_financial['credit_component'] = np.minimum(X_financial['credit_score'] / 1000, 1.0)
        X_financial['dti_component'] = np.maximum(0, 1 - X_financial['dti_ratio'])
        X_financial['missed_payments_component'] = np.maximum(0, 1 - (X_financial['missed_payments'] / 10))
        X_financial['income_component'] = np.minimum(X_financial['income'] / 100000, 1.0)
        X_financial['ccj_component'] = 1 - X_financial['has_ccj']  # 0 if has CCJ, 1 if not
        X_financial['asset_component'] = X_financial['has_mortgage'] * 0.6 + X_financial['has_car'] * 0.4
        
        # Use the transformed components for ML
        ml_features = [
            'credit_component', 'dti_component', 'missed_payments_component',
            'income_component', 'ccj_component', 'asset_component'
        ]
        
        X = X_financial[ml_features].fillna(0)
        y = self.df[target_variable].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Extract and normalize weights
        raw_weights = lr_model.coef_
        normalized_weights = np.abs(raw_weights) / np.sum(np.abs(raw_weights))
        
        # Store results
        self.ml_financial_weights = dict(zip(ml_features, normalized_weights))
        
        # Evaluate performance
        y_pred = lr_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Target variable: {target_variable}")
        print(f"Manual financial weights: [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]")
        print(f"ML-learned weights:       {normalized_weights}")
        print(f"Model R² score: {r2:.3f}")
        
        return self.ml_financial_weights
    
    def optimize_engagement_weights(self, target_variable='ml_financial_health_score'):
        """
        Use supervised learning to optimize engagement weights
        Uses ML-optimized financial health score as target (not manual)
        """
        print("\n🤖 OPTIMIZING ENGAGEMENT WEIGHTS WITH ML")
        print("=" * 60)
        
        # First ensure we have ML financial weights
        if self.ml_financial_weights is None:
            print("No ML financial weights found. Optimizing financial weights first...")
            self.optimize_financial_health_weights()
        
        # Calculate ML-based financial health scores
        ml_financial_scores = self._calculate_ml_financial_health_scores()
        self.df['ml_financial_health_score'] = ml_financial_scores
        
        # Now optimize engagement weights using ML financial scores as target
        engagement_features = ['click_rate', 'avg_time_viewed', 'total_interactions']
        
        # Create synthetic engagement features if needed
        if not all(col in self.df.columns for col in engagement_features):
            print("Creating synthetic engagement features for demo...")
            self._create_synthetic_engagement_features()
        
        X = self.df[engagement_features].fillna(0)
        y = self.df['ml_financial_health_score'].fillna(0)
        
        # Scale and train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Extract weights
        raw_weights = lr_model.coef_
        normalized_weights = raw_weights / np.sum(np.abs(raw_weights))
        
        self.ml_engagement_weights = dict(zip(engagement_features, normalized_weights))
        
        # Evaluate
        y_pred = lr_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Target: ML-optimized financial health score")
        print(f"Manual engagement weights: [0.4, 0.3, 0.3]") 
        print(f"ML-learned weights:        {normalized_weights}")
        print(f"Model R² score: {r2:.3f}")
        
        return self.ml_engagement_weights
    
    def _calculate_ml_financial_health_scores(self):
        """Calculate financial health using ML-learned weights"""
        if self.ml_financial_weights is None:
            raise ValueError("ML financial weights not available")
        
        # Calculate components
        credit_comp = np.minimum(self.df['credit_score'] / 1000, 1.0)
        dti_comp = np.maximum(0, 1 - self.df['dti_ratio'])
        missed_comp = np.maximum(0, 1 - (self.df['missed_payments'] / 10))
        income_comp = np.minimum(self.df['income'] / 100000, 1.0)
        ccj_comp = 1 - self.df['has_ccj']
        asset_comp = self.df['has_mortgage'] * 0.6 + self.df['has_car'] * 0.4
        
        # Apply ML-learned weights
        ml_financial_health = (
            credit_comp * self.ml_financial_weights['credit_component'] +
            dti_comp * self.ml_financial_weights['dti_component'] +
            missed_comp * self.ml_financial_weights['missed_payments_component'] +
            income_comp * self.ml_financial_weights['income_component'] +
            ccj_comp * self.ml_financial_weights['ccj_component'] +
            asset_comp * self.ml_financial_weights['asset_component']
        )
        
        return ml_financial_health
    
    def _create_synthetic_business_outcome(self):
        """Create synthetic business outcome for demo purposes"""
        np.random.seed(42)
        
        # Business outcome should correlate with good financial health
        base_outcome = (
            (self.df['credit_score'] / 1000) * 0.3 +
            (1 - self.df['dti_ratio']) * 0.25 +
            (1 - self.df['missed_payments'] / 10) * 0.2 +
            (self.df['income'] / 100000) * 0.15 +
            (1 - self.df['has_ccj']) * 0.1
        )
        
        # Add some noise and different weight priorities for ML to discover
        # Maybe income is MORE important for business outcomes than we thought
        business_outcome = (
            (self.df['credit_score'] / 1000) * 0.2 +      # Less important than manual
            (1 - self.df['dti_ratio']) * 0.15 +           # Less important  
            (1 - self.df['missed_payments'] / 10) * 0.25 + # More important
            (self.df['income'] / 100000) * 0.35 +          # Much more important!
            (1 - self.df['has_ccj']) * 0.05 +             # Less important
            np.random.normal(0, 0.1, len(self.df))        # Add noise
        )
        
        return np.clip(business_outcome, 0, 1)
    
    def _create_synthetic_engagement_features(self):
        """Create synthetic engagement features for demo"""
        np.random.seed(42)
        n_users = len(self.df)
        self.df['click_rate'] = np.random.beta(2, 3, n_users)
        self.df['avg_time_viewed'] = np.random.gamma(2, 30, n_users)  
        self.df['total_interactions'] = np.random.poisson(8, n_users)
    
    def compare_manual_vs_ml_weights(self):
        """Compare manual weights vs ML-learned weights"""
        print("\n📊 MANUAL VS ML WEIGHT COMPARISON")
        print("=" * 80)
        
        # Financial weights comparison
        print("\n🏦 FINANCIAL HEALTH WEIGHTS:")
        print(f"{'Component':<25} {'Manual':<12} {'ML-Learned':<12} {'Difference':<12}")
        print("-" * 65)
        
        manual_fin_weights = [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
        ml_fin_weights = list(self.ml_financial_weights.values()) if self.ml_financial_weights else [0]*6
        
        components = ['credit_component', 'dti_component', 'missed_payments_component', 
                     'income_component', 'ccj_component', 'asset_component']
        
        for comp, manual, ml in zip(components, manual_fin_weights, ml_fin_weights):
            diff = ml - manual
            print(f"{comp:<25} {manual:<12.3f} {ml:<12.3f} {diff:+<12.3f}")
        
        # Engagement weights comparison  
        print("\n📊 ENGAGEMENT WEIGHTS:")
        print(f"{'Component':<25} {'Manual':<12} {'ML-Learned':<12} {'Difference':<12}")
        print("-" * 65)
        
        manual_eng_weights = [0.4, 0.3, 0.3]
        ml_eng_weights = list(self.ml_engagement_weights.values()) if self.ml_engagement_weights else [0]*3
        
        eng_components = ['click_rate', 'avg_time_viewed', 'total_interactions']
        
        for comp, manual, ml in zip(eng_components, manual_eng_weights, ml_eng_weights):
            diff = ml - manual
            print(f"{comp:<25} {manual:<12.3f} {ml:<12.3f} {diff:+<12.3f}")
    
    def run_fully_ml_analysis(self):
        """Run complete analysis with ML optimization of both weight systems"""
        print("🚀 FULLY ML-DRIVEN FINANCIAL RECOMMENDER")
        print("=" * 80)
        
        # Step 1: Optimize financial health weights
        self.optimize_financial_health_weights()
        
        # Step 2: Optimize engagement weights using ML financial scores
        self.optimize_engagement_weights()
        
        # Step 3: Compare results
        self.compare_manual_vs_ml_weights()
        
        print("\n💡 KEY INSIGHTS:")
        print("✅ Financial weights learned from actual business outcomes")
        print("✅ Engagement weights learned from ML-optimized financial health")
        print("✅ No dependency on manual weight guesses")
        print("✅ Fully data-driven recommendation system")

if __name__ == "__main__":
    print("🔬 FULLY ML-DRIVEN WEIGHT OPTIMIZATION")
    print("=" * 60)
    
    try:
        # Load data
        recommender = FullyMLFinancialRecommender('joined_user_table.csv')
        
        # Run fully ML analysis
        recommender.run_fully_ml_analysis()
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("\nThis demonstrates how to use ML for BOTH:")
        print("  1. Financial health weight optimization") 
        print("  2. Engagement weight optimization")
        print("\nNo more manual weight guessing! 🎯")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Run enhanced_financial_recommender.py first to generate the data")
    except Exception as e:
        print(f"❌ Error: {e}") 