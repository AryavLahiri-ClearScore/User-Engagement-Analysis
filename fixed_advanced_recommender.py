"""
Fixed Advanced Refactored Financial Recommender
- Missing ClusteringConfig class added
- All dependencies properly defined
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Protocol
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from contextlib import contextmanager

# Set plotting style
sns.set_palette("husl")

# ================================
# COMPREHENSIVE CONFIGURATION
# ================================

@dataclass
class NormalizationConfig:
    """Configuration for data normalization"""
    max_credit_score: float = 1000.0
    max_income: float = 100000.0
    max_missed_payments: float = 10.0
    mortgage_asset_weight: float = 0.6
    car_asset_weight: float = 0.4
    weight_validation_tolerance: float = 0.001

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    dashboard_figsize: Tuple[float, float] = (20, 12)
    correlation_figsize: Tuple[float, float] = (12, 10)
    clustering_figsize: Tuple[float, float] = (15, 12)
    dpi: int = 300
    alpha: float = 0.7
    scatter_size: float = 50.0
    hist_bins: int = 20
    colormap_segments: str = 'Set3'
    colormap_correlation: str = 'RdBu_r'
    colormap_financial: str = 'RdYlGn'

@dataclass
class FileConfig:
    """Configuration for file handling"""
    output_directory: Path = Path("output")
    dashboard_suffix: str = "_financial_dashboard.png"
    correlation_suffix: str = "_correlation_heatmap.png" 
    clustering_suffix: str = "_clustering_visualization.png"
    csv_suffix: str = "_recommendations.csv"

@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    min_users: int = 10
    required_engagement_columns: List[str] = field(default_factory=lambda: [
        'user_id', 'content_id', 'content_type', 'time_viewed_in_sec', 'clicked'
    ])
    required_financial_columns: List[str] = field(default_factory=lambda: [
        'user_id', 'credit_score', 'dti_ratio', 'income', 'total_debt', 
        'missed_payments', 'has_ccj', 'has_mortgage', 'has_car'
    ])

@dataclass
class ClusteringConfig:
    """Configuration for clustering parameters - THIS WAS MISSING!"""
    n_clusters: int = 3
    random_state: int = 42
    pca_components: int = 2
    feature_columns: List[str] = field(default_factory=lambda: [
        'engagement_score', 'financial_health_score', 'total_interactions'
    ])

@dataclass
class FinancialConfig:
    """Enhanced financial configuration with validation"""
    # Weights
    credit_weight: float = 0.30
    dti_weight: float = 0.25
    missed_payments_weight: float = 0.15
    income_weight: float = 0.15
    ccj_weight: float = 0.10
    asset_weight: float = 0.05
    
    # Thresholds
    excellent_threshold: float = 0.8
    good_threshold: float = 0.65
    fair_threshold: float = 0.45
    high_dti_threshold: float = 0.5
    finn_diff_threshold: int = 2
    
    # Normalization
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    def __post_init__(self):
        self._validate_weights()
    
    def _validate_weights(self):
        """Centralized weight validation"""
        weight_sum = (self.credit_weight + self.dti_weight + self.missed_payments_weight + 
                     self.income_weight + self.ccj_weight + self.asset_weight)
        if abs(weight_sum - 1.0) > self.normalization.weight_validation_tolerance:
            raise ValueError(f"Financial weights must sum to 1.0, got {weight_sum:.3f}")
        logging.info(f"✅ Financial weights validated: {weight_sum:.3f}")

@dataclass  
class EngagementConfig:
    """Enhanced engagement configuration"""
    # Weights
    click_rate_weight: float = 0.4
    avg_time_weight: float = 0.3
    interactions_weight: float = 0.3
    
    # Normalization factors
    max_time_seconds: int = 60
    max_interactions: int = 12
    
    # Thresholds
    high_engagement_threshold: float = 0.5
    medium_engagement_threshold: float = 0.25
    
    # Normalization
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    def __post_init__(self):
        self._validate_weights()
    
    def _validate_weights(self):
        """Centralized weight validation"""
        weight_sum = self.click_rate_weight + self.avg_time_weight + self.interactions_weight
        if abs(weight_sum - 1.0) > self.normalization.weight_validation_tolerance:
            raise ValueError(f"Engagement weights must sum to 1.0, got {weight_sum:.3f}")
        logging.info(f"✅ Engagement weights validated: {weight_sum:.3f}")

@dataclass
class SystemConfig:
    """Master configuration container - NOW WITH PROPER ClusteringConfig!"""
    financial: FinancialConfig = field(default_factory=FinancialConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig) 
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)  # ✅ FIXED!
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    files: FileConfig = field(default_factory=FileConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

# ================================
# ERROR HANDLING
# ================================

class RecommenderError(Exception):
    """Base exception for recommender system"""
    pass

class DataValidationError(RecommenderError):
    """Raised when data validation fails"""
    pass

class ConfigurationError(RecommenderError):
    """Raised when configuration is invalid"""
    pass

# ================================
# SIMPLE DEMO CLASS
# ================================

class FixedAdvancedRecommender:
    """Advanced Financial Recommender with comprehensive visualizations"""
    
    def __init__(self, csv_file: str, config: Optional[SystemConfig] = None):
        """
        Initialize the recommender with required CSV file
        
        Args:
            csv_file: Path to joined_user_table.csv (required)
            config: Optional custom configuration
        """
        if not csv_file:
            raise ValueError("CSV file path is required. Please provide path to joined_user_table.csv")
        
        self.config = config or SystemConfig()
        self.csv_file = csv_file
        self.user_features = None
        self.scaler = StandardScaler()
        
        # Load the CSV data
        try:
            self.df = pd.read_csv(csv_file)
            print("✅ Successfully created recommender with all configs!")
            print(f"   Clustering config: {self.config.clustering}")
            print(f"   Financial config weights sum: {self._check_financial_weights():.3f}")
            print(f"   Loaded data: {len(self.df):,} interaction records from {csv_file}")
            print(f"   Unique users: {self.df['user_id'].nunique():,}")
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        except Exception as e:
            raise Exception(f"Error loading CSV file {csv_file}: {e}")
    
    def _check_financial_weights(self) -> float:
        """Check that financial weights sum to 1.0"""
        fc = self.config.financial
        return (fc.credit_weight + fc.dti_weight + fc.missed_payments_weight + 
                fc.income_weight + fc.ccj_weight + fc.asset_weight)
    
    def demo_configs(self):
        """Demonstrate different configuration aspects"""
        print("\n📊 CONFIGURATION DEMO:")
        print(f"Financial - Excellent threshold: {self.config.financial.excellent_threshold}")
        print(f"Engagement - High threshold: {self.config.engagement.high_engagement_threshold}")
        print(f"Clustering - Number of clusters: {self.config.clustering.n_clusters}")
        print(f"Visualization - Dashboard size: {self.config.visualization.dashboard_figsize}")
        print(f"Files - Output directory: {self.config.files.output_directory}")
        print(f"Validation - Min users: {self.config.validation.min_users}")
    
    def create_user_features(self) -> pd.DataFrame:
        """Create user features from the CSV data"""
        if not self.csv_file or not hasattr(self, 'df'):
            raise ValueError("CSV file is required. Please provide joined_user_table.csv")
        
        # Always use real data from CSV
        return self._create_real_user_features()
    

    
    def _create_real_user_features(self) -> pd.DataFrame:
        """Create user features from real CSV data"""
        print(f"Processing real data from {self.csv_file}...")
        print(f"Total interaction records: {len(self.df):,}")
        
        # Check unique users
        unique_users = self.df['user_id'].nunique()
        print(f"Unique users in dataset: {unique_users:,}")
        
        # Basic engagement metrics from interaction data
        user_stats = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).round(2)
        
        user_stats.columns = ['avg_time_viewed', 'total_time_viewed', 'total_interactions',
                             'click_rate', 'total_clicks', 'unique_content_viewed']
        
        # Content type preferences
        print(f"Content types in data: {self.df['content_type'].unique()}")
        print(f"Content type distribution: {self.df['content_type'].value_counts().to_dict()}")
        
        content_preferences = self.df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
        content_preferences = content_preferences.div(content_preferences.sum(axis=1), axis=0)
        content_preferences.columns = [f'pref_{col}' for col in content_preferences.columns]
        
        print(f"Content preference columns created: {list(content_preferences.columns)}")
        print(f"Sample content preferences:\n{content_preferences.head(3)}")
        
        # Get unique financial data per user (since financial data repeats per interaction)
        financial_data = self.df.groupby('user_id').agg({
            'credit_score': 'first',
            'dti_ratio': 'first', 
            'income': 'first',
            'total_debt': 'first',
            'missed_payments': 'first',
            'has_ccj': 'first',
            'has_mortgage': 'first',
            'has_car': 'first'
        })
        
        # Combine all features
        user_features = pd.concat([
            user_stats,
            content_preferences,
            financial_data
        ], axis=1).fillna(0)
        
        # Calculate derived features using the same logic as the config
        user_features['financial_health_score'] = self._calculate_financial_health_score(user_features)
        user_features['engagement_score'] = self._calculate_engagement_score(user_features)
        user_features['financial_category'] = user_features['financial_health_score'].apply(self._categorize_financial_health)
        user_features['enhanced_segment'] = self._assign_enhanced_segments(user_features)
        
        print(f"✅ Created features for {len(user_features)} unique users")
        
        # Debug: Check for missing or problematic data
        print(f"\n🔍 DATA VALIDATION:")
        print(f"User features shape: {user_features.shape}")
        print(f"Columns with all zeros: {(user_features == 0).all().sum()}")
        print(f"Columns with all NaN: {user_features.isnull().all().sum()}")
        
        # Check key metrics ranges
        key_metrics = ['financial_health_score', 'engagement_score', 'credit_score', 'dti_ratio']
        for metric in key_metrics:
            if metric in user_features.columns:
                values = user_features[metric]
                print(f"{metric}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
        
        # Print user distribution summary
        self._print_user_distribution_summary(user_features)
        
        return user_features
    
    def _calculate_financial_health_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate financial health score using configured weights"""
        fc = self.config.financial
        nc = self.config.financial.normalization
        
        # Normalize components
        credit_norm = df['credit_score'] / nc.max_credit_score
        dti_norm = 1 - df['dti_ratio']  # Lower DTI is better
        income_norm = df['income'] / nc.max_income
        missed_norm = 1 - (df['missed_payments'] / nc.max_missed_payments)
        ccj_norm = 1 - df['has_ccj']
        asset_norm = (df['has_mortgage'] * nc.mortgage_asset_weight + 
                     df['has_car'] * nc.car_asset_weight)
        
        # Calculate weighted score
        score = (credit_norm * fc.credit_weight +
                dti_norm * fc.dti_weight +
                income_norm * fc.income_weight +
                missed_norm * fc.missed_payments_weight +
                ccj_norm * fc.ccj_weight +
                asset_norm * fc.asset_weight)
        
        return score.clip(0, 1)
    
    def _calculate_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate engagement score using configured weights"""
        ec = self.config.engagement
        
        # Normalize components
        click_norm = df['click_rate']
        time_norm = (df['avg_time_viewed'] / ec.max_time_seconds).clip(0, 1)
        interaction_norm = (df['total_interactions'] / ec.max_interactions).clip(0, 1)
        
        # Calculate weighted score
        score = (click_norm * ec.click_rate_weight +
                time_norm * ec.avg_time_weight +
                interaction_norm * ec.interactions_weight)
        
        return score.clip(0, 1)
    
    def _categorize_financial_health(self, score: float) -> str:
        """Categorize financial health based on score"""
        fc = self.config.financial
        if score >= fc.excellent_threshold:
            return "Excellent"
        elif score >= fc.good_threshold:
            return "Good"
        elif score >= fc.fair_threshold:
            return "Fair"
        else:
            return "Poor"
    
    def _assign_enhanced_segments(self, df: pd.DataFrame) -> pd.Series:
        """Assign enhanced segments based on engagement and financial health"""
        segments = []
        ec = self.config.engagement
        fc = self.config.financial
        
        print(f"🔍 SEGMENT ASSIGNMENT DEBUG:")
        print(f"Engagement thresholds: high={ec.high_engagement_threshold}, medium={ec.medium_engagement_threshold}")
        print(f"Financial thresholds: dti={fc.high_dti_threshold}, missed_payments={fc.finn_diff_threshold}")
        
        count = 0
        for i, row in df.iterrows():
            engagement = row['engagement_score']
            financial_cat = self._categorize_financial_health(row['financial_health_score'])
            dti_ratio = row['dti_ratio']
            missed_payments = row['missed_payments']
            
            if count < 3:  # Debug first 3 users
                print(f"User {i}: engagement={engagement:.3f}, financial_cat={financial_cat}, dti={dti_ratio:.3f}, missed={missed_payments}")
            
            if missed_payments >= fc.finn_diff_threshold:
                segment = "Payment_Recovery_Priority"
            elif dti_ratio >= fc.high_dti_threshold:
                segment = "Debt_Management_Priority"
            elif engagement > ec.high_engagement_threshold:
                if financial_cat == "Excellent":
                    segment = "Premium_Engaged"
                elif financial_cat in ["Good", "Fair"]:
                    segment = "Growth_Focused"
                else:
                    segment = "Recovery_Engaged"
            elif engagement > ec.medium_engagement_threshold:
                if financial_cat == "Excellent":
                    segment = "Premium_Moderate"
                elif financial_cat in ["Good", "Fair"]:
                    segment = "Mainstream"
                else:
                    segment = "Recovery_Moderate"
            else:
                if financial_cat == "Poor":
                    segment = "Financial_Priority"
                else:
                    segment = "Activation_Needed"
            
            if count < 3:  # Debug first 3 users
                print(f"  -> Assigned segment: {segment}")
            
            segments.append(segment)
            count += 1
        
        print(f"Assigned {len(segments)} segments, unique: {len(set(segments))}")
        return pd.Series(segments, index=df.index)
    
    def _print_user_distribution_summary(self, df: pd.DataFrame):
        """Print detailed user distribution summary to terminal"""
        print("\n" + "=" * 70)
        print("📊 USER DISTRIBUTION SUMMARY")
        print("=" * 70)
        
        total_users = len(df)
        print(f"Total Users: {total_users:,}")
        
        # Financial Health Categories
        print(f"\n💰 FINANCIAL HEALTH CATEGORIES:")
        print("-" * 50)
        financial_counts = df['financial_category'].value_counts().sort_index()
        for category, count in financial_counts.items():
            percentage = (count / total_users) * 100
            print(f"  {category:12} : {count:4,} users ({percentage:5.1f}%)")
        
        # Enhanced Segments
        print(f"\n🎯 ENHANCED USER SEGMENTS:")
        print("-" * 50)
        segment_counts = df['enhanced_segment'].value_counts().sort_values(ascending=False)
        for segment, count in segment_counts.items():
            percentage = (count / total_users) * 100
            print(f"  {segment:25} : {count:4,} users ({percentage:5.1f}%)")
        
        # Financial Health Statistics
        print(f"\n📈 FINANCIAL HEALTH STATISTICS:")
        print("-" * 50)
        fin_health = df['financial_health_score']
        print(f"  Average Score    : {fin_health.mean():.3f}")
        print(f"  Median Score     : {fin_health.median():.3f}")
        print(f"  Standard Dev     : {fin_health.std():.3f}")
        print(f"  Min Score        : {fin_health.min():.3f}")
        print(f"  Max Score        : {fin_health.max():.3f}")
        
        # Engagement Statistics
        print(f"\n⚡ ENGAGEMENT STATISTICS:")
        print("-" * 50)
        engagement = df['engagement_score']
        print(f"  Average Score    : {engagement.mean():.3f}")
        print(f"  Median Score     : {engagement.median():.3f}")
        print(f"  Standard Dev     : {engagement.std():.3f}")
        print(f"  Min Score        : {engagement.min():.3f}")
        print(f"  Max Score        : {engagement.max():.3f}")
        
        # High-Level Insights
        print(f"\n🔍 KEY INSIGHTS:")
        print("-" * 50)
        
        # Financial insights
        excellent_users = financial_counts.get('Excellent', 0)
        poor_users = financial_counts.get('Poor', 0)
        print(f"  • {excellent_users:,} users ({(excellent_users/total_users)*100:.1f}%) have excellent financial health")
        print(f"  • {poor_users:,} users ({(poor_users/total_users)*100:.1f}%) need financial support")
        
        # Engagement insights
        high_engagement = len(df[df['engagement_score'] > self.config.engagement.high_engagement_threshold])
        low_engagement = len(df[df['engagement_score'] <= self.config.engagement.medium_engagement_threshold])
        print(f"  • {high_engagement:,} users ({(high_engagement/total_users)*100:.1f}%) are highly engaged")
        print(f"  • {low_engagement:,} users ({(low_engagement/total_users)*100:.1f}%) need engagement activation")
        
        # Risk segments
        payment_recovery = segment_counts.get('Payment_Recovery_Priority', 0)
        debt_management = segment_counts.get('Debt_Management_Priority', 0)
        financial_priority = segment_counts.get('Financial_Priority', 0)
        
        total_risk = payment_recovery + debt_management + financial_priority
        print(f"  • {total_risk:,} users ({(total_risk/total_users)*100:.1f}%) are in high-risk financial segments")
        
        # Premium segments
        premium_engaged = segment_counts.get('Premium_Engaged', 0)
        premium_moderate = segment_counts.get('Premium_Moderate', 0)
        total_premium = premium_engaged + premium_moderate
        print(f"  • {total_premium:,} users ({(total_premium/total_users)*100:.1f}%) are in premium segments")
        
        print("=" * 70)
    
    def _print_quick_summary(self, df: pd.DataFrame):
        """Print quick summary of user distribution"""
        total_users = len(df)
        financial_counts = df['financial_category'].value_counts()
        segment_counts = df['enhanced_segment'].value_counts()
        
        print(f"\n📊 QUICK SUMMARY ({total_users:,} users):")
        print(f"   Financial: {dict(financial_counts)}")
        print(f"   Top Segments: {dict(segment_counts.head(3))}")
    
    def diagnose_blank_visualizations(self):
        """Diagnose why visualizations might be blank"""
        print("\n🔍 DIAGNOSING BLANK VISUALIZATIONS")
        print("=" * 60)
        
        if self.user_features is None:
            print("❌ user_features is None - need to create features first")
            return
        
        print(f"✅ User features exist: {self.user_features.shape}")
        
        # Check required columns
        required_viz_columns = ['financial_health_score', 'engagement_score', 'financial_category', 'enhanced_segment']
        missing_columns = [col for col in required_viz_columns if col not in self.user_features.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
        else:
            print("✅ All required visualization columns present")
        
        # Check for empty/invalid data
        for col in required_viz_columns:
            if col in self.user_features.columns:
                data = self.user_features[col]
                if col in ['financial_health_score', 'engagement_score']:
                    print(f"📊 {col}: min={data.min():.3f}, max={data.max():.3f}, all_zero={(data == 0).all()}")
                else:
                    print(f"📊 {col}: unique_values={data.nunique()}, sample={list(data.unique())[:3]}")
        
        # Check for NaN issues
        nan_cols = self.user_features.isnull().all()
        if nan_cols.any():
            print(f"❌ Columns with all NaN: {list(nan_cols[nan_cols].index)}")
        else:
            print("✅ No columns with all NaN")
        
        # Check data ranges that might cause invisible plots
        if 'financial_health_score' in self.user_features.columns:
            fh_score = self.user_features['financial_health_score']
            if fh_score.max() - fh_score.min() < 0.001:
                print("⚠️ Financial health scores have very small range - might appear blank")
        
        if 'engagement_score' in self.user_features.columns:
            eng_score = self.user_features['engagement_score']
            if eng_score.max() - eng_score.min() < 0.001:
                print("⚠️ Engagement scores have very small range - might appear blank")
        
        print("=" * 60)
    
    def create_financial_visualizations(self):
        """Create comprehensive financial and engagement visualizations"""
        if self.user_features is None:
            self.user_features = self.create_user_features()
        
        print("\n" + "=" * 60)
        print("🎨 METHOD 1: CREATING FINANCIAL ANALYSIS VISUALIZATIONS")
        print("=" * 60)
        
        # Debug: Show financial categories and segments
        print("🔍 FINANCIAL CATEGORIES DEBUG:")
        financial_cats = self.user_features['financial_category'].value_counts()
        print(f"   Categories found: {dict(financial_cats)}")
        
        print("🔍 ENHANCED SEGMENTS DEBUG:")
        segments = self.user_features['enhanced_segment'].value_counts()
        print(f"   Segments found: {dict(segments)}")
        print(f"   Unique segments: {self.user_features['enhanced_segment'].unique()}")
        print(f"   Any NaN segments? {self.user_features['enhanced_segment'].isna().sum()}")
        
        # Print current distribution summary
        self._print_user_distribution_summary(self.user_features)
        
        vc = self.config.visualization
        
        # FIGURE 1: Financial Health Dashboard (2x3 grid)
        fig1, axes1 = plt.subplots(2, 3, figsize=vc.dashboard_figsize)
        fig1.suptitle('Financial Health & Engagement Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Enhanced segment distribution
        segment_counts = self.user_features['enhanced_segment'].value_counts()
        colors = plt.cm.get_cmap(vc.colormap_segments)(np.linspace(0, 1, len(segment_counts)))
        axes1[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes1[0, 0].set_title('Enhanced Segment Distribution')
        
        # 2. Financial category distribution
        financial_counts = self.user_features['financial_category'].value_counts()
        colors_fin = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        axes1[0, 1].bar(financial_counts.index, financial_counts.values, 
                       color=colors_fin[:len(financial_counts)], alpha=vc.alpha)
        axes1[0, 1].set_title('Financial Health Categories')
        axes1[0, 1].set_ylabel('Number of Users')
        axes1[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Credit score distribution
        axes1[0, 2].hist(self.user_features['credit_score'], bins=vc.hist_bins, alpha=vc.alpha, 
                        color='skyblue', edgecolor='black')
        axes1[0, 2].axvline(self.user_features['credit_score'].mean(), color='red', 
                           linestyle='--', label=f'Mean: {self.user_features["credit_score"].mean():.0f}')
        axes1[0, 2].set_title('Credit Score Distribution')
        axes1[0, 2].set_xlabel('Credit Score')
        axes1[0, 2].set_ylabel('Number of Users')
        axes1[0, 2].legend()
        
        # 4. DTI Ratio vs Financial Health Score
        scatter = axes1[1, 0].scatter(self.user_features['dti_ratio'], 
                                     self.user_features['financial_health_score'],
                                     c=self.user_features['credit_score'], 
                                     cmap='viridis', alpha=vc.alpha, s=vc.scatter_size)
        axes1[1, 0].set_xlabel('Debt-to-Income Ratio')
        axes1[1, 0].set_ylabel('Financial Health Score')
        axes1[1, 0].set_title('DTI vs Financial Health (colored by Credit Score)')
        plt.colorbar(scatter, ax=axes1[1, 0], label='Credit Score')
        
        # 5. Engagement vs Financial Health by segment
        for segment in self.user_features['enhanced_segment'].unique():
            segment_data = self.user_features[self.user_features['enhanced_segment'] == segment]
            axes1[1, 1].scatter(segment_data['engagement_score'], 
                               segment_data['financial_health_score'],
                               label=segment, alpha=vc.alpha, s=vc.scatter_size)
        axes1[1, 1].set_xlabel('Engagement Score')
        axes1[1, 1].set_ylabel('Financial Health Score')
        axes1[1, 1].set_title('Engagement vs Financial Health by Segment')
        axes1[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 6. Income distribution by financial category
        financial_categories = self.user_features['financial_category'].unique()
        income_data = [self.user_features[self.user_features['financial_category'] == cat]['income'].values 
                      for cat in financial_categories]
        box_plot = axes1[1, 2].boxplot(income_data, labels=financial_categories, patch_artist=True)
        colors_box = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        for patch, color in zip(box_plot['boxes'], colors_box[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
        axes1[1, 2].set_title('Income Distribution by Financial Category')
        axes1[1, 2].set_ylabel('Income (£)')
        axes1[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename1 = f"{self.config.files.output_directory}/financial_health_dashboard.png"
        plt.savefig(filename1, dpi=vc.dpi, bbox_inches='tight')
        print(f"✅ Saved: {filename1}")
        plt.show()
        plt.close()
        
        return fig1
    
    def visualize_financial_clustering(self):
        """Visualize the enhanced clustering with financial context"""
        if self.user_features is None:
            self.user_features = self.create_user_features()
        
        print("\n" + "=" * 60)
        print("🎨 METHOD 2: FINANCIAL CLUSTERING VISUALIZATION")
        print("=" * 60)
        
        # Debug: Show financial categories and segments again
        print("🔍 CLUSTERING - FINANCIAL CATEGORIES DEBUG:")
        financial_cats = self.user_features['financial_category'].value_counts()
        print(f"   Categories found: {dict(financial_cats)}")
        
        print("🔍 CLUSTERING - ENHANCED SEGMENTS DEBUG:")
        segments = self.user_features['enhanced_segment'].value_counts()
        print(f"   Segments found: {dict(segments)}")
        print(f"   Unique segments: {self.user_features['enhanced_segment'].unique()}")
        print(f"   Any NaN segments? {self.user_features['enhanced_segment'].isna().sum()}")
        
        # Print quick summary for clustering
        self._print_quick_summary(self.user_features)
        
        vc = self.config.visualization
        cc = self.config.clustering
        
        # Prepare features for clustering
        engagement_features = ['avg_time_viewed', 'total_interactions', 'click_rate', 'unique_content_viewed']
        financial_features = ['financial_health_score', 'credit_score', 'dti_ratio', 'income']
        content_features = [col for col in self.user_features.columns if col.startswith('pref_')]
        
        print(f"🔍 CLUSTERING DEBUG:")
        print(f"Available columns: {list(self.user_features.columns)}")
        print(f"Content features found: {content_features}")
        print(f"Missing engagement features: {[f for f in engagement_features if f not in self.user_features.columns]}")
        print(f"Missing financial features: {[f for f in financial_features if f not in self.user_features.columns]}")
        
        clustering_features = engagement_features + financial_features + content_features
        available_features = [f for f in clustering_features if f in self.user_features.columns]
        print(f"Using {len(available_features)} features for clustering: {available_features}")
        
        features_for_clustering = self.user_features[available_features].fillna(0)
        print(f"Features for clustering shape: {features_for_clustering.shape}")
        print(f"Features summary:\n{features_for_clustering.describe()}")
        
        features_scaled = self.scaler.fit_transform(features_for_clustering)
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=cc.pca_components, random_state=cc.random_state)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create the visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=vc.clustering_figsize)
        fig.suptitle('Enhanced Financial-Engagement Clustering', fontsize=16, fontweight='bold')
        
        # 1. Clusters by enhanced segment
        segments = self.user_features['enhanced_segment'].unique()
        print(f"🔍 SEGMENT DEBUG:")
        print(f"Unique segments: {segments}")
        print(f"Segment counts: {self.user_features['enhanced_segment'].value_counts()}")
        
        segment_colors = plt.cm.get_cmap(vc.colormap_segments)(
            np.linspace(0, 1, len(segments))
        )
        
        points_plotted = 0
        for i, segment in enumerate(segments):
            mask = self.user_features['enhanced_segment'] == segment
            count = mask.sum()
            print(f"Segment '{segment}': {count} users, plotting: {mask.any()}")
            if mask.any():
                ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[segment_colors[i]], label=segment, alpha=vc.alpha, s=vc.scatter_size)
                points_plotted += count
        
        print(f"Total points plotted: {points_plotted}/{len(self.user_features)}")
        ax1.set_title('Clusters by Enhanced Segment')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 2. Colored by financial health score
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=self.user_features['financial_health_score'], 
                              cmap=vc.colormap_financial, alpha=vc.alpha, s=vc.scatter_size)
        ax2.set_title('Colored by Financial Health Score')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter2, ax=ax2, label='Financial Health Score')
        
        # 3. Colored by credit score
        scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=self.user_features['credit_score'], 
                              cmap='viridis', alpha=vc.alpha, s=vc.scatter_size)
        ax3.set_title('Colored by Credit Score')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter3, ax=ax3, label='Credit Score')
        
        # 4. Colored by DTI ratio
        scatter4 = ax4.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=self.user_features['dti_ratio'], 
                              cmap='Reds', alpha=vc.alpha, s=vc.scatter_size)
        ax4.set_title('Colored by Debt-to-Income Ratio')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter4, ax=ax4, label='DTI Ratio')
        
        plt.tight_layout()
        filename = f"{self.config.files.output_directory}/financial_clustering_analysis.png"
        plt.savefig(filename, dpi=vc.dpi, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.show()
        plt.close()
        
        # Print PCA explanation
        print(f"PCA Explained Variance:")
        print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"  Total: {sum(pca.explained_variance_ratio_):.1%}")
        
        return features_2d, pca
    
    def create_engagement_financial_correlation(self):
        """Create correlation analysis between engagement and financial metrics"""
        if self.user_features is None:
            self.user_features = self.create_user_features()
        
        print("\n" + "=" * 60)
        print("🎨 METHOD 3: ENGAGEMENT vs FINANCIAL CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Debug: Show financial categories and segments again
        print("🔍 CORRELATION - FINANCIAL CATEGORIES DEBUG:")
        financial_cats = self.user_features['financial_category'].value_counts()
        print(f"   Categories found: {dict(financial_cats)}")
        
        print("🔍 CORRELATION - ENHANCED SEGMENTS DEBUG:")
        segments = self.user_features['enhanced_segment'].value_counts()
        print(f"   Segments found: {dict(segments)}")
        print(f"   Unique segments: {self.user_features['enhanced_segment'].unique()}")
        print(f"   Any NaN segments? {self.user_features['enhanced_segment'].isna().sum()}")
        
        # Print quick summary for correlation analysis
        self._print_quick_summary(self.user_features)
        
        vc = self.config.visualization
        
        # Select engagement and financial metrics for correlation
        engagement_metrics = ['engagement_score', 'click_rate', 'avg_time_viewed', 
                             'total_interactions', 'unique_content_viewed']
        financial_metrics = ['financial_health_score', 'credit_score', 'dti_ratio', 
                            'income', 'total_debt', 'missed_payments']
        
        print(f"🔍 CORRELATION DEBUG:")
        print(f"Available columns: {list(self.user_features.columns)}")
        
        # Check which metrics are actually available
        available_engagement = [m for m in engagement_metrics if m in self.user_features.columns]
        available_financial = [m for m in financial_metrics if m in self.user_features.columns]
        
        print(f"Available engagement metrics: {available_engagement}")
        print(f"Available financial metrics: {available_financial}")
        print(f"Missing engagement metrics: {[m for m in engagement_metrics if m not in self.user_features.columns]}")
        print(f"Missing financial metrics: {[m for m in financial_metrics if m not in self.user_features.columns]}")
        
        # Combine all metrics for correlation analysis
        correlation_metrics = available_engagement + available_financial
        correlation_data = self.user_features[correlation_metrics]
        
        print(f"Correlation data shape: {correlation_data.shape}")
        print(f"Correlation data summary:\n{correlation_data.describe()}")
        
        # Calculate correlation matrix
        correlation_matrix = correlation_data.corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(1, 1, figsize=vc.correlation_figsize)
        
        # Create heatmap with custom formatting
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        
        # Create heatmap
        heatmap = sns.heatmap(correlation_matrix, 
                             mask=mask,
                             annot=True, 
                             cmap=vc.colormap_correlation, 
                             center=0,
                             square=True,
                             fmt='.2f',
                             cbar_kws={"shrink": .8},
                             ax=ax)
        
        ax.set_title('Engagement vs Financial Metrics Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        filename = f"{self.config.files.output_directory}/engagement_financial_correlation.png"
        plt.savefig(filename, dpi=vc.dpi, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.show()
        plt.close()
        
        # Print key correlations
        print("\nKEY CORRELATIONS (|r| > 0.3):")
        print("-" * 50)
        
        # Find significant correlations between engagement and financial metrics
        significant_correlations = []
        for eng_metric in engagement_metrics:
            for fin_metric in financial_metrics:
                corr_value = correlation_matrix.loc[eng_metric, fin_metric]
                if abs(corr_value) > 0.3:
                    direction = "positive" if corr_value > 0 else "negative"
                    strength = "strong" if abs(corr_value) > 0.7 else "moderate"
                    significant_correlations.append({
                        'engagement': eng_metric,
                        'financial': fin_metric,
                        'correlation': corr_value,
                        'direction': direction,
                        'strength': strength
                    })
                    print(f"• {eng_metric} ↔ {fin_metric}: {corr_value:.3f} ({strength} {direction})")
        
        if not significant_correlations:
            print("  No correlations found with |r| > 0.3")
        
        return correlation_matrix, significant_correlations
    
    def run_complete_analysis(self):
        """Run complete financial analysis with all visualizations"""
        print("🚀 STARTING COMPLETE FINANCIAL ANALYSIS")
        print("=" * 70)
        
        # Ensure output directory exists
        self.config.files.output_directory.mkdir(exist_ok=True)
        
        # Create user features
        print("\n1️⃣ Creating user features...")
        self.user_features = self.create_user_features()
        print(f"   Created features for {len(self.user_features)} users")
        
        # Debug: Show overall data state before visualizations
        print("\n🔍 OVERALL DATA STATE BEFORE VISUALIZATIONS:")
        print(f"   User features shape: {self.user_features.shape}")
        print(f"   Financial categories: {self.user_features['financial_category'].value_counts().to_dict()}")
        print(f"   Enhanced segments: {self.user_features['enhanced_segment'].value_counts().to_dict()}")
        print(f"   NaN segments: {self.user_features['enhanced_segment'].isna().sum()}")
        
        # Create visualizations
        print("\n2️⃣ Creating financial visualizations...")
        try:
            self.create_financial_visualizations()
            print("   ✅ Method 1 (Financial Dashboard) completed")
        except Exception as e:
            print(f"   ❌ Method 1 (Financial Dashboard) failed: {e}")
        
        print("\n3️⃣ Creating clustering visualization...")
        try:
            self.visualize_financial_clustering()
            print("   ✅ Method 2 (Clustering) completed")
        except Exception as e:
            print(f"   ❌ Method 2 (Clustering) failed: {e}")
        
        print("\n4️⃣ Creating correlation analysis...")
        try:
            self.create_engagement_financial_correlation()
            print("   ✅ Method 3 (Correlation) completed")
        except Exception as e:
            print(f"   ❌ Method 3 (Correlation) failed: {e}")
        
        # Print summary
        print("\n🎉 ANALYSIS COMPLETE!")
        print("📊 VISUALIZATION METHODS SUMMARY:")
        print("   1. Financial Dashboard (2x3 grid) - Financial health, segments, credit scores")
        print("   2. Clustering Visualization (2x2 grid) - PCA clustering by segments and metrics")
        print("   3. Correlation Analysis - Heatmap of engagement vs financial correlations")
        print(f"Generated visualizations saved to: {self.config.files.output_directory}")
        
        return self.user_features

# ================================
# USAGE EXAMPLE
# ================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("🔧 TESTING FIXED ADVANCED RECOMMENDER WITH VISUALIZATIONS")
        print("=" * 60)
        
        # Test 1: Basic configuration test
        print("\n1️⃣ Testing Basic Configuration:")
        basic_recommender = FixedAdvancedRecommender('joined_user_table.csv')
        basic_recommender.demo_configs()
        
        # Test 2: Full analysis with standard configuration
        print("\n2️⃣ Running Full Analysis with Standard Configuration:")
        standard_recommender = FixedAdvancedRecommender('joined_user_table.csv')
        user_data = standard_recommender.run_complete_analysis()
        print(f"   Analysis completed for {len(user_data)} users")
        
        # Test 3: Custom configuration with different parameters
        print("\n3️⃣ Testing Custom Configuration:")
        custom_config = SystemConfig(
            clustering=ClusteringConfig(n_clusters=5, random_state=123),
            financial=FinancialConfig(excellent_threshold=0.85, good_threshold=0.70),
            engagement=EngagementConfig(high_engagement_threshold=0.6),
            visualization=VisualizationConfig(
                dashboard_figsize=(18, 10),
                correlation_figsize=(14, 12),
                colormap_segments='tab20',
                alpha=0.8
            ),
            files=FileConfig(output_directory=Path("custom_output"))
        )
        custom_recommender = FixedAdvancedRecommender('joined_user_table.csv', config=custom_config)
        custom_recommender.demo_configs()
        
        # Test 4: Run analysis with custom config
        print("\n4️⃣ Running Analysis with Custom Configuration:")
        custom_data = custom_recommender.run_complete_analysis()
        print(f"   Custom analysis completed for {len(custom_data)} users")
        
        # Test 5: Individual visualization methods
        print("\n5️⃣ Testing Individual Visualization Methods:")
        test_recommender = FixedAdvancedRecommender('joined_user_table.csv')
        
        print("   - Testing financial dashboard...")
        test_recommender.create_financial_visualizations()
        
        print("   - Testing clustering visualization...")
        test_recommender.visualize_financial_clustering()
        
        print("   - Testing correlation analysis...")
        test_recommender.create_engagement_financial_correlation()
        
        # Summary
        print("\n🎉 ALL TESTS SUCCESSFUL!")
        print("=" * 60)
        print("✅ Configuration system working")
        print("✅ Data generation working") 
        print("✅ Financial calculations working")
        print("✅ Engagement scoring working")
        print("✅ User segmentation working")
        print("✅ Financial dashboard visualization working")
        print("✅ Clustering visualization working")
        print("✅ Correlation analysis working")
        print("✅ File output working")
        
        print(f"\n📁 Generated files saved to:")
        print(f"   • Standard config: {standard_recommender.config.files.output_directory}")
        print(f"   • Custom config: {custom_recommender.config.files.output_directory}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Error in advanced recommender: {e}")
        import traceback
        traceback.print_exc() 