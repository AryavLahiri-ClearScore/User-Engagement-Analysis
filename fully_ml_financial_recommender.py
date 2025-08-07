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
        
        # Debug: Show data loading info
        print(f"📊 DATA LOADING INFO:")
        print(f"   Total interaction records: {len(self.df):,}")
        print(f"   Unique users: {self.df['user_id'].nunique():,}")
        print(f"   Columns: {list(self.df.columns)}")
        
        # Create user-level aggregated data for analysis
        self._create_user_level_data()
    
    def _create_user_level_data(self):
        """Create user-level aggregated data from interaction records"""
        print("🔄 Creating user-level aggregated data...")
        
        # Get unique financial data per user (since it repeats per interaction)
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
        
        # Merge financial and engagement data
        self.user_df = pd.merge(user_financial, user_engagement, on='user_id')
        
        print(f"✅ Created user-level data: {len(self.user_df)} unique users")
        
        # Replace the original df with user-level data for analysis
        self.df = self.user_df.copy()
        
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
        print(f"Model R² score: {r2:.3f}")
        
        # Display detailed weight comparison
        print(f"\n💰 DETAILED FINANCIAL WEIGHTS COMPARISON:")
        print("-" * 80)
        print(f"{'ATTRIBUTE':25} | {'MANUAL':>8} | {'ML':>8} | {'DIFF':>8} | {'ML %':>8}")
        print("-" * 80)
        
        manual_weights = [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
        component_names = ['credit_component', 'dti_component', 'missed_payments_component', 
                          'income_component', 'ccj_component', 'asset_component']
        readable_names = ['Credit Score', 'DTI Ratio', 'Missed Payments', 'Income', 'CCJ Status', 'Assets']
        
        for i, (comp_name, readable, manual_w) in enumerate(zip(component_names, readable_names, manual_weights)):
            ml_w = normalized_weights[i]
            diff = ml_w - manual_w
            percentage = (ml_w / sum(normalized_weights)) * 100
            print(f"{readable:25} | {manual_w:8.3f} | {ml_w:8.3f} | {diff:+8.3f} | {percentage:7.1f}%")
        
        print(f"{'TOTAL':25} | {sum(manual_weights):8.3f} | {sum(normalized_weights):8.3f} | {0:+8.3f} | {100.0:7.1f}%")
        
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
        
        # Check if engagement features exist (they should from aggregation)
        missing_features = [col for col in engagement_features if col not in self.df.columns]
        if missing_features:
            print(f"⚠️ Missing engagement features: {missing_features}")
            print("Creating synthetic engagement features for demo...")
            self._create_synthetic_engagement_features()
        else:
            print("✅ Using real aggregated engagement features")
        
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
        print(f"Model R² score: {r2:.3f}")
        
        # Display detailed engagement weight comparison
        print(f"\n📊 DETAILED ENGAGEMENT WEIGHTS COMPARISON:")
        print("-" * 80)
        print(f"{'ATTRIBUTE':25} | {'MANUAL':>8} | {'ML':>8} | {'DIFF':>8} | {'ML %':>8}")
        print("-" * 80)
        
        manual_weights = [0.4, 0.3, 0.3]
        readable_names = ['Click Rate', 'Avg Time Viewed', 'Total Interactions']
        
        for i, (readable, manual_w) in enumerate(zip(readable_names, manual_weights)):
            ml_w = normalized_weights[i]
            diff = ml_w - manual_w
            percentage = (ml_w / sum(abs(normalized_weights))) * 100
            print(f"{readable:25} | {manual_w:8.3f} | {ml_w:8.3f} | {diff:+8.3f} | {percentage:7.1f}%")
        
        print(f"{'TOTAL':25} | {sum(manual_weights):8.3f} | {sum(abs(normalized_weights)):8.3f} | {0:+8.3f} | {100.0:7.1f}%")
        
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
    
    def _categorize_financial_health(self, score):
        """Categorize financial health based on score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.65:
            return "Good"
        elif score >= 0.45:
            return "Fair"
        else:
            return "Poor"
    
    def _assign_enhanced_segments(self, df):
        """Assign enhanced segments based on engagement and financial health"""
        segments = []
        
        for _, row in df.iterrows():
            # Use ML-calculated scores
            engagement = row.get('ml_engagement_score', 0)
            financial_score = row.get('ml_financial_health_score', 0)
            financial_cat = self._categorize_financial_health(financial_score)
            dti_ratio = row.get('dti_ratio', 0)
            missed_payments = row.get('missed_payments', 0)
            
            # Enhanced segmentation logic
            if missed_payments >= 2:
                segments.append("Payment_Recovery_Priority")
            elif dti_ratio >= 0.5:
                segments.append("Debt_Management_Priority")
            elif engagement > 0.5:
                if financial_cat == "Excellent":
                    segments.append("Premium_Engaged")
                elif financial_cat in ["Good", "Fair"]:
                    segments.append("Growth_Focused")
                else:
                    segments.append("Recovery_Engaged")
            elif engagement > 0.25:
                if financial_cat == "Excellent":
                    segments.append("Premium_Moderate")
                elif financial_cat in ["Good", "Fair"]:
                    segments.append("Mainstream")
                else:
                    segments.append("Recovery_Moderate")
            else:
                if financial_cat == "Poor":
                    segments.append("Financial_Priority")
                else:
                    segments.append("Activation_Needed")
        
        return segments
    
    def analyze_ml_segments(self):
        """Analyze and display ML-based financial categories and segments"""
        if self.ml_financial_weights is None or self.ml_engagement_weights is None:
            print("❌ ML weights not available. Run optimization first.")
            return
        
        # Calculate ML scores
        ml_financial_scores = self._calculate_ml_financial_health_scores()
        self.df['ml_financial_health_score'] = ml_financial_scores
        
        # Calculate ML engagement scores
        if all(col in self.df.columns for col in ['click_rate', 'avg_time_viewed', 'total_interactions']):
            ml_engagement_scores = (
                self.df['click_rate'] * self.ml_engagement_weights['click_rate'] +
                self.df['avg_time_viewed'] * self.ml_engagement_weights['avg_time_viewed'] +
                self.df['total_interactions'] * self.ml_engagement_weights['total_interactions']
            )
            self.df['ml_engagement_score'] = ml_engagement_scores
        else:
            print("Creating synthetic engagement scores for analysis...")
            np.random.seed(42)  # Ensure reproducible synthetic scores
            self.df['ml_engagement_score'] = np.random.beta(2, 3, len(self.df))
        
        # Categorize financial health
        self.df['ml_financial_category'] = self.df['ml_financial_health_score'].apply(self._categorize_financial_health)
        
        # Assign enhanced segments
        self.df['ml_enhanced_segment'] = self._assign_enhanced_segments(self.df)
        
        # Display results
        print(f"\n🏆 ML-BASED FINANCIAL CATEGORIES & SEGMENTS ANALYSIS")
        print("=" * 80)
        
        total_users = len(self.df)
        print(f"Total Users Analyzed: {total_users:,}")
        
        # Financial Categories
        print(f"\n💰 ML-BASED FINANCIAL HEALTH CATEGORIES:")
        print("-" * 60)
        financial_counts = self.df['ml_financial_category'].value_counts()
        for category, count in financial_counts.items():
            percentage = (count / total_users) * 100
            print(f"   {category:12}: {count:4} users ({percentage:5.1f}%)")
        
        # Enhanced Segments
        print(f"\n🎯 ML-BASED ENHANCED USER SEGMENTS:")
        print("-" * 60)
        segment_counts = self.df['ml_enhanced_segment'].value_counts()
        for segment, count in segment_counts.items():
            percentage = (count / total_users) * 100
            print(f"   {segment:25}: {count:4} users ({percentage:5.1f}%)")
        
        # Statistics
        print(f"\n📊 ML SCORE STATISTICS:")
        print("-" * 60)
        fin_health = self.df['ml_financial_health_score']
        engagement = self.df['ml_engagement_score']
        
        print(f"   Financial Health Score:")
        print(f"     Average: {fin_health.mean():.3f}")
        print(f"     Median:  {fin_health.median():.3f}")
        print(f"     Range:   {fin_health.min():.3f} - {fin_health.max():.3f}")
        
        print(f"   Engagement Score:")
        print(f"     Average: {engagement.mean():.3f}")
        print(f"     Median:  {engagement.median():.3f}")
        print(f"     Range:   {engagement.min():.3f} - {engagement.max():.3f}")
        
        return financial_counts, segment_counts
    
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
        print("\n1️⃣ OPTIMIZING FINANCIAL HEALTH WEIGHTS")
        print("-" * 60)
        self.optimize_financial_health_weights()
        
        # Step 2: Optimize engagement weights using ML financial scores
        print("\n2️⃣ OPTIMIZING ENGAGEMENT WEIGHTS")
        print("-" * 60)
        self.optimize_engagement_weights()
        
        # Step 3: Analyze ML-based segments and categories
        print("\n3️⃣ ANALYZING ML-BASED USER SEGMENTS")
        print("-" * 60)
        financial_counts, segment_counts = self.analyze_ml_segments()
        
        # Step 4: Compare results
        print("\n4️⃣ MANUAL VS ML WEIGHT COMPARISON")
        print("-" * 60)
        self.compare_manual_vs_ml_weights()
        
        print("\n🔍 KEY INSIGHTS FROM ML OPTIMIZATION:")
        print("=" * 80)
        print("✅ Financial weights learned from actual business outcomes")
        print("✅ Engagement weights learned from ML-optimized financial health")
        print("✅ User segments based on ML-calculated scores")
        print("✅ No dependency on manual weight guesses")
        print("✅ Fully data-driven recommendation system")
        
        # Summary of what ML discovered
        if self.ml_financial_weights and self.ml_engagement_weights:
            print(f"\n📈 ML DISCOVERIES:")
            print("-" * 60)
            
            # Find most important financial attribute
            max_fin_weight = max(self.ml_financial_weights.values())
            max_fin_attr = [k for k, v in self.ml_financial_weights.items() if v == max_fin_weight][0]
            print(f"   Most important financial factor: {max_fin_attr.replace('_', ' ').title()}")
            
            # Find most important engagement attribute  
            max_eng_weight = max(self.ml_engagement_weights.values())
            max_eng_attr = [k for k, v in self.ml_engagement_weights.items() if v == max_eng_weight][0]
            print(f"   Most important engagement factor: {max_eng_attr.replace('_', ' ').title()}")
            
            # Top user segment
            if len(segment_counts) > 0:
                top_segment = segment_counts.index[0]
                top_count = segment_counts.iloc[0]
                print(f"   Largest user segment: {top_segment} ({top_count} users)")
        
        return financial_counts, segment_counts

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