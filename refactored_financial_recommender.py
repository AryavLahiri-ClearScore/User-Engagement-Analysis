"""
Refactored Financial Recommender with improved architecture
- Separation of concerns
- Configuration management  
- Modular design
- Better testability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

# ================================
# CONFIGURATION MANAGEMENT
# ================================

@dataclass
class FinancialConfig:
    """Centralized configuration for financial health scoring"""
    # Financial health weights
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
    
    def __post_init__(self):
        """Validate that weights sum to 1.0"""
        weight_sum = (self.credit_weight + self.dti_weight + self.missed_payments_weight + 
                     self.income_weight + self.ccj_weight + self.asset_weight)
        if abs(weight_sum - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Financial weights must sum to 1.0, got {weight_sum:.3f}")
        print(f"✅ Financial weights validated: {weight_sum:.3f}")

@dataclass  
class EngagementConfig:
    """Configuration for engagement scoring"""
    click_rate_weight: float = 0.4
    avg_time_weight: float = 0.3
    interactions_weight: float = 0.3
    
    # Normalization factors
    max_time_seconds: int = 60
    max_interactions: int = 12
    
    # Engagement thresholds
    high_engagement_threshold: float = 0.5
    medium_engagement_threshold: float = 0.25
    
    def __post_init__(self):
        """Validate that weights sum to 1.0"""
        weight_sum = self.click_rate_weight + self.avg_time_weight + self.interactions_weight
        if abs(weight_sum - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Engagement weights must sum to 1.0, got {weight_sum:.3f}")
        print(f"✅ Engagement weights validated: {weight_sum:.3f}")

@dataclass
class ClusteringConfig:
    """Configuration for clustering"""
    n_clusters: int = 6
    random_state: int = 42
    n_init: int = 10

# ================================
# CORE DATA MODELS
# ================================

@dataclass
class UserFinancialProfile:
    """Represents a user's financial profile"""
    user_id: str
    credit_score: float
    dti_ratio: float
    income: float
    total_debt: float
    missed_payments: int
    has_ccj: bool
    has_mortgage: bool
    has_car: bool
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'credit_score': self.credit_score,
            'dti_ratio': self.dti_ratio,
            'income': self.income,
            'total_debt': self.total_debt,
            'missed_payments': self.missed_payments,
            'has_ccj': self.has_ccj,
            'has_mortgage': self.has_mortgage,
            'has_car': self.has_car
        }

@dataclass
class UserEngagementProfile:
    """Represents a user's engagement profile"""
    user_id: str
    click_rate: float
    avg_time_viewed: float
    total_interactions: int
    unique_content_viewed: int
    content_preferences: Dict[str, float]

# ================================
# FINANCIAL HEALTH CALCULATOR
# ================================

class FinancialHealthCalculator:
    """Handles financial health score calculations"""
    
    def __init__(self, config: FinancialConfig):
        self.config = config
    
    def calculate_components(self, profile: UserFinancialProfile) -> Dict[str, float]:
        """Calculate individual financial components"""
        return {
            'credit_component': min(profile.credit_score / 1000, 1.0),
            'dti_component': max(0, 1 - profile.dti_ratio),
            'missed_payments_component': max(0, 1 - (profile.missed_payments / 10)),
            'income_component': min(profile.income / 100000, 1.0),
            'ccj_component': 0.0 if profile.has_ccj else 1.0,
            'asset_component': (profile.has_mortgage * 0.6 + profile.has_car * 0.4)
        }
    
    def calculate_health_score(self, profile: UserFinancialProfile) -> float:
        """Calculate composite financial health score"""
        components = self.calculate_components(profile)
        
        return (
            components['credit_component'] * self.config.credit_weight +
            components['dti_component'] * self.config.dti_weight +
            components['missed_payments_component'] * self.config.missed_payments_weight +
            components['income_component'] * self.config.income_weight +
            components['ccj_component'] * self.config.ccj_weight +
            components['asset_component'] * self.config.asset_weight
        )
    
    def categorize_health(self, health_score: float) -> str:
        """Categorize financial health based on score"""
        if health_score >= self.config.excellent_threshold:
            return "Excellent"
        elif health_score >= self.config.good_threshold:
            return "Good"
        elif health_score >= self.config.fair_threshold:
            return "Fair"
        else:
            return "Poor"

# ================================
# ENGAGEMENT CALCULATOR
# ================================

class EngagementCalculator:
    """Handles engagement score calculations"""
    
    def __init__(self, config: EngagementConfig):
        self.config = config
    
    def calculate_engagement_score(self, profile: UserEngagementProfile) -> float:
        """Calculate composite engagement score"""
        normalized_time = min(profile.avg_time_viewed / self.config.max_time_seconds, 1)
        normalized_interactions = min(profile.total_interactions / self.config.max_interactions, 1)
        
        return (
            profile.click_rate * self.config.click_rate_weight +
            normalized_time * self.config.avg_time_weight +
            normalized_interactions * self.config.interactions_weight
        )

# ================================
# SEGMENTATION STRATEGY
# ================================

class SegmentationStrategy(ABC):
    """Abstract base class for segmentation strategies"""
    
    @abstractmethod
    def assign_segment(self, engagement_score: float, financial_profile: UserFinancialProfile) -> str:
        pass

class PriorityBasedSegmentation(SegmentationStrategy):
    """Priority-based segmentation with DTI and payment issues taking precedence"""
    
    def __init__(self, financial_config: FinancialConfig, engagement_config: EngagementConfig):
        self.financial_config = financial_config
        self.engagement_config = engagement_config
    
    def assign_segment(self, engagement_score: float, financial_profile: UserFinancialProfile) -> str:
        """Assign segment based on priority rules"""
        # Priority 1: Payment issues
        if financial_profile.missed_payments >= self.financial_config.finn_diff_threshold:
            return "Payment_Recovery_Priority"
        
        # Priority 2: High DTI
        elif financial_profile.dti_ratio >= self.financial_config.high_dti_threshold:
            return "Debt_Management_Priority"
        
        # Standard engagement-based segmentation
        return self._assign_engagement_segment(engagement_score, financial_profile)
    
    def _assign_engagement_segment(self, engagement_score: float, financial_profile: UserFinancialProfile) -> str:
        """Assign segment based on engagement level using configurable thresholds"""
        # Calculate financial health for context
        calculator = FinancialHealthCalculator(self.financial_config)
        health_score = calculator.calculate_health_score(financial_profile)
        financial_category = calculator.categorize_health(health_score)
        
        # Use configurable engagement thresholds instead of magic numbers
        if engagement_score > self.engagement_config.high_engagement_threshold:  # Was: 0.5
            if financial_category == "Excellent":
                return "Premium_Engaged"
            elif financial_category in ["Good", "Fair"]:
                return "Growth_Focused"
            else:
                return "Recovery_Engaged"
        elif engagement_score > self.engagement_config.medium_engagement_threshold:  # Was: 0.25
            if financial_category == "Excellent":
                return "Premium_Moderate"
            elif financial_category in ["Good", "Fair"]:
                return "Mainstream"
            else:
                return "Recovery_Moderate"
        else:
            if financial_category == "Poor":
                return "Financial_Priority"
            else:
                return "Activation_Needed"

# ================================
# VISUALIZATION FACTORY (COMPLETE IMPLEMENTATION)
# ================================

class VisualizationFactory:
    """Factory for creating different types of visualizations"""
    
    @staticmethod
    def create_financial_dashboard(user_features: pd.DataFrame) -> plt.Figure:
        """Create comprehensive financial health dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Refactored Financial Health & Engagement Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Enhanced segment distribution
        segment_counts = user_features['enhanced_segment'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[0, 0].set_title('Enhanced Segment Distribution')
        
        # 2. Financial category distribution
        financial_counts = user_features['financial_category'].value_counts()
        colors_fin = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        axes[0, 1].bar(financial_counts.index, financial_counts.values, color=colors_fin[:len(financial_counts)])
        axes[0, 1].set_title('Financial Health Categories')
        axes[0, 1].set_ylabel('Number of Users')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Credit score distribution
        axes[0, 2].hist(user_features['credit_score'], bins=20, alpha=0.7, 
                        color='skyblue', edgecolor='black')
        axes[0, 2].axvline(user_features['credit_score'].mean(), color='red', 
                           linestyle='--', label=f'Mean: {user_features["credit_score"].mean():.0f}')
        axes[0, 2].set_title('Credit Score Distribution')
        axes[0, 2].set_xlabel('Credit Score')
        axes[0, 2].set_ylabel('Number of Users')
        axes[0, 2].legend()
        
        # 4. DTI Ratio vs Financial Health Score
        scatter = axes[1, 0].scatter(user_features['dti_ratio'], 
                                     user_features['financial_health_score'],
                                     c=user_features['credit_score'], 
                                     cmap='viridis', alpha=0.7, s=50)
        axes[1, 0].set_xlabel('Debt-to-Income Ratio')
        axes[1, 0].set_ylabel('Financial Health Score')
        axes[1, 0].set_title('DTI vs Financial Health (colored by Credit Score)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Credit Score')
        
        # 5. Engagement vs Financial Health by segment
        for segment in user_features['enhanced_segment'].unique():
            segment_data = user_features[user_features['enhanced_segment'] == segment]
            axes[1, 1].scatter(segment_data['engagement_score'], 
                               segment_data['financial_health_score'],
                               label=segment, alpha=0.7, s=50)
        axes[1, 1].set_xlabel('Engagement Score')
        axes[1, 1].set_ylabel('Financial Health Score')
        axes[1, 1].set_title('Engagement vs Financial Health by Segment')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 6. Income distribution by financial category
        financial_categories = user_features['financial_category'].unique()
        income_data = [user_features[user_features['financial_category'] == cat]['income'].values 
                      for cat in financial_categories]
        box_plot = axes[1, 2].boxplot(income_data, labels=financial_categories, patch_artist=True)
        colors_box = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        for patch, color in zip(box_plot['boxes'], colors_box[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
        axes[1, 2].set_title('Income Distribution by Financial Category')
        axes[1, 2].set_ylabel('Income (£)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_correlation_heatmap(user_features: pd.DataFrame) -> plt.Figure:
        """Create engagement vs financial correlation heatmap"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Select metrics for correlation
        engagement_metrics = ['engagement_score', 'click_rate', 'avg_time_viewed', 'total_interactions']
        financial_metrics = ['financial_health_score', 'credit_score', 'dti_ratio', 'income', 'missed_payments']
        
        # Combine metrics
        correlation_metrics = engagement_metrics + financial_metrics
        correlation_data = user_features[correlation_metrics]
        
        # Calculate correlation matrix
        correlation_matrix = correlation_data.corr()
        
        # Create heatmap with mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        
        heatmap = sns.heatmap(correlation_matrix, 
                             mask=mask,
                             annot=True, 
                             cmap='RdBu_r', 
                             center=0,
                             square=True,
                             fmt='.2f',
                             cbar_kws={"shrink": .8},
                             ax=ax)
        
        ax.set_title('Engagement vs Financial Metrics Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_clustering_visualization(user_features: pd.DataFrame, clustering_config: ClusteringConfig) -> plt.Figure:
        """Create PCA clustering visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Refactored Financial-Engagement Clustering', fontsize=16, fontweight='bold')
        
        # Prepare features for clustering
        engagement_features = ['avg_time_viewed', 'total_interactions', 'click_rate', 'unique_content_viewed']
        financial_features = ['financial_health_score', 'credit_score', 'dti_ratio', 'income']
        content_features = [col for col in user_features.columns if col.startswith('pref_')]
        
        clustering_features = engagement_features + financial_features + content_features
        features_for_clustering = user_features[clustering_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_clustering)
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=clustering_config.random_state)
        features_2d = pca.fit_transform(features_scaled)
        
        # 1. Clusters by enhanced segment
        segment_colors = plt.cm.Set3(np.linspace(0, 1, len(user_features['enhanced_segment'].unique())))
        segments = user_features['enhanced_segment'].unique()
        
        for i, segment in enumerate(segments):
            mask = user_features['enhanced_segment'] == segment
            if mask.any():
                ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[segment_colors[i]], label=segment, alpha=0.7, s=50)
        ax1.set_title('Clusters by Enhanced Segment')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 2. Colored by financial health score
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=user_features['financial_health_score'], 
                              cmap='RdYlGn', alpha=0.7, s=50)
        ax2.set_title('Colored by Financial Health Score')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter2, ax=ax2, label='Financial Health Score')
        
        # 3. Colored by credit score
        scatter3 = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=user_features['credit_score'], 
                              cmap='viridis', alpha=0.7, s=50)
        ax3.set_title('Colored by Credit Score')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter3, ax=ax3, label='Credit Score')
        
        # 4. Colored by DTI ratio
        scatter4 = ax4.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=user_features['dti_ratio'], 
                              cmap='Reds', alpha=0.7, s=50)
        ax4.set_title('Colored by Debt-to-Income Ratio')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter4, ax=ax4, label='DTI Ratio')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def display_and_save_figure(fig: plt.Figure, filename: str, show: bool = True):
        """Helper method to save and optionally display figures"""
        # Save figure
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        
        # Display if requested
        if show:
            plt.show()  # 🖼️ THIS MAKES THE VISUALIZATION POP UP!
        
        # Close to free memory
        plt.close(fig)

# ================================
# MAIN RECOMMENDER CLASS (REFACTORED)
# ================================

class RefactoredFinancialRecommender:
    """Refactored financial recommender with improved architecture"""
    
    def __init__(self, 
                 csv_file: str,
                 financial_config: Optional[FinancialConfig] = None,
                 engagement_config: Optional[EngagementConfig] = None,
                 clustering_config: Optional[ClusteringConfig] = None):
        
        self.df = pd.read_csv(csv_file)
        self.financial_config = financial_config or FinancialConfig()
        self.engagement_config = engagement_config or EngagementConfig()
        self.clustering_config = clustering_config or ClusteringConfig()
        
        # Components
        self.financial_calculator = FinancialHealthCalculator(self.financial_config)
        self.engagement_calculator = EngagementCalculator(self.engagement_config)
        self.segmentation_strategy = PriorityBasedSegmentation(self.financial_config, self.engagement_config)
        
        # Data
        self.user_features = None
        self.scaler = StandardScaler()
    
    def create_user_features(self) -> pd.DataFrame:
        """Create enhanced user features with better separation of concerns"""
        print("Creating enhanced user features...")
        
        # Get engagement features
        engagement_features = self._create_engagement_features()
        
        # Get financial features  
        financial_features = self._create_financial_features()
        
        # Get content preferences
        content_features = self._create_content_preferences()
        
        # Combine all features
        self.user_features = pd.concat([
            engagement_features,
            content_features, 
            financial_features
        ], axis=1).fillna(0)
        
        return self.user_features
    
    def _create_engagement_features(self) -> pd.DataFrame:
        """Create engagement-related features"""
        engagement_stats = self.df.groupby('user_id').agg({
            'time_viewed_in_sec': ['mean', 'sum', 'count'],
            'clicked': ['mean', 'sum'],
            'content_id': 'nunique'
        }).round(2)
        
        engagement_stats.columns = [
            'avg_time_viewed', 'total_time_viewed', 'total_interactions',
            'click_rate', 'total_clicks', 'unique_content_viewed'
        ]
        
        return engagement_stats
    
    def _create_financial_features(self) -> pd.DataFrame:
        """Create financial health features"""
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
        
        return user_financial
    
    def _create_content_preferences(self) -> pd.DataFrame:
        """Create content preference features"""
        content_preferences = self.df.groupby(['user_id', 'content_type']).size().unstack(fill_value=0)
        content_preferences = content_preferences.div(content_preferences.sum(axis=1), axis=0)
        content_preferences.columns = [f'pref_{col}' for col in content_preferences.columns]
        
        return content_preferences
    
    def perform_segmentation(self) -> pd.DataFrame:
        """Perform user segmentation with configurable strategy"""
        print("Performing enhanced segmentation...")
        
        if self.user_features is None:
            self.create_user_features()
        
        # Calculate engagement scores
        engagement_scores = []
        segments = []
        
        for user_id, user_data in self.user_features.iterrows():
            # Create engagement profile
            eng_profile = UserEngagementProfile(
                user_id=user_id,
                click_rate=user_data['click_rate'],
                avg_time_viewed=user_data['avg_time_viewed'],
                total_interactions=user_data['total_interactions'],
                unique_content_viewed=user_data['unique_content_viewed'],
                content_preferences={}  # Could be populated from pref_ columns
            )
            
            # Create financial profile
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
            
            # Calculate scores and assign segments
            engagement_score = self.engagement_calculator.calculate_engagement_score(eng_profile)
            segment = self.segmentation_strategy.assign_segment(engagement_score, fin_profile)
            
            engagement_scores.append(engagement_score)
            segments.append(segment)
        
        self.user_features['engagement_score'] = engagement_scores
        self.user_features['enhanced_segment'] = segments
        
        print("Enhanced segment distribution:")
        print(self.user_features['enhanced_segment'].value_counts())
        
        return self.user_features
    
    def analyze_segments(self) -> Dict:
        """Analyze segment characteristics"""
        if 'enhanced_segment' not in self.user_features.columns:
            self.perform_segmentation()
        
        segment_analysis = self.user_features.groupby('enhanced_segment').agg({
            'financial_health_score': ['mean', 'std'],
            'credit_score': ['mean', 'min', 'max'],
            'dti_ratio': ['mean', 'std'],
            'income': ['mean', 'median'],
            'engagement_score': ['mean']
        }).round(2)
        
        return segment_analysis
    
    def run_analysis(self, config_name: str = "standard") -> Tuple[pd.DataFrame, Dict]:
        """Run complete analysis pipeline with configurable naming"""
        print(f"🚀 STARTING REFACTORED FINANCIAL ANALYSIS ({config_name.upper()})")
        print("=" * 70)
        
        # Create features
        features = self.create_user_features()
        
        # Perform segmentation  
        segmented_features = self.perform_segmentation()
        
        # Analyze results
        analysis = self.analyze_segments()
        
        # Create visualizations with unique names
        print(f"\n📊 CREATING {config_name.upper()} VISUALIZATIONS...")
        financial_dashboard_fig = VisualizationFactory.create_financial_dashboard(segmented_features)
        correlation_heatmap_fig = VisualizationFactory.create_correlation_heatmap(segmented_features)
        clustering_visualization_fig = VisualizationFactory.create_clustering_visualization(segmented_features, self.clustering_config)
        
        # Display and save visualizations with unique filenames
        VisualizationFactory.display_and_save_figure(
            financial_dashboard_fig, f"{config_name}_financial_dashboard.png"
        )
        VisualizationFactory.display_and_save_figure(
            correlation_heatmap_fig, f"{config_name}_correlation_heatmap.png"
        )
        VisualizationFactory.display_and_save_figure(
            clustering_visualization_fig, f"{config_name}_clustering_visualization.png"
        )
        
        print(f"\n✅ {config_name.upper()} ANALYSIS COMPLETE!")
        return segmented_features, analysis

# ================================
# CONFIGURATION EXAMPLES
# ================================

def create_conservative_config() -> Tuple[FinancialConfig, EngagementConfig]:
    """Create conservative configuration with stricter thresholds"""
    financial_config = FinancialConfig(
        # ALL weights specified to sum to 1.0 (conservative emphasis on credit/debt)
        credit_weight=0.35,           # +0.05 from default (more conservative)
        dti_weight=0.30,              # +0.05 from default (more conservative)  
        missed_payments_weight=0.15,  # Same as default
        income_weight=0.10,           # -0.05 from default (less emphasis)
        ccj_weight=0.05,              # -0.05 from default (less emphasis)
        asset_weight=0.05,            # Same as default
        # Total: 0.35 + 0.30 + 0.15 + 0.10 + 0.05 + 0.05 = 1.00 ✅
        
        high_dti_threshold=0.4,       # Lower DTI threshold (stricter)
        excellent_threshold=0.85,     # Higher bar for "Excellent"
        good_threshold=0.70           # Higher bar for "Good"
    )
    
    engagement_config = EngagementConfig(
        # ALL weights specified to sum to 1.0 (conservative emphasis on clicks)
        click_rate_weight=0.5,        # +0.1 from default (more emphasis on clicks)
        avg_time_weight=0.25,         # -0.05 from default
        interactions_weight=0.25,     # -0.05 from default
        # Total: 0.5 + 0.25 + 0.25 = 1.00 ✅
        
        high_engagement_threshold=0.6,   # Stricter high engagement threshold
        medium_engagement_threshold=0.35 # Higher medium threshold
    )
    
    return financial_config, engagement_config

def create_aggressive_config() -> Tuple[FinancialConfig, EngagementConfig]:
    """Create aggressive growth configuration with more lenient thresholds"""
    financial_config = FinancialConfig(
        # ALL weights specified to sum to 1.0 (aggressive emphasis on income/assets)
        credit_weight=0.20,           # -0.10 from default (less conservative)
        dti_weight=0.20,              # -0.05 from default (less conservative)
        missed_payments_weight=0.10,  # -0.05 from default (more forgiving)
        income_weight=0.25,           # +0.10 from default (growth potential)
        ccj_weight=0.05,              # -0.05 from default (more forgiving)
        asset_weight=0.20,            # +0.15 from default (wealth indicators)
        # Total: 0.20 + 0.20 + 0.10 + 0.25 + 0.05 + 0.20 = 1.00 ✅
        
        high_dti_threshold=0.6,       # Higher DTI tolerance (more lenient)
        excellent_threshold=0.75,     # Lower bar for "Excellent"
        good_threshold=0.60           # Lower bar for "Good"
    )
    
    engagement_config = EngagementConfig(
        # ALL weights specified to sum to 1.0 (aggressive emphasis on volume)
        click_rate_weight=0.3,        # -0.1 from default
        avg_time_weight=0.3,          # Same as default
        interactions_weight=0.4,      # +0.1 from default (volume focus)
        # Total: 0.3 + 0.3 + 0.4 = 1.00 ✅
        
        high_engagement_threshold=0.4,   # More lenient high engagement
        medium_engagement_threshold=0.15 # Lower medium threshold
    )
    
    return financial_config, engagement_config

def create_balanced_config() -> Tuple[FinancialConfig, EngagementConfig]:
    """Create balanced configuration for A/B testing"""
    financial_config = FinancialConfig(
        # ALL weights specified to sum to 1.0 (equal emphasis approach)
        credit_weight=0.20,           # Equal weight
        dti_weight=0.20,              # Equal weight
        missed_payments_weight=0.20,  # Equal weight
        income_weight=0.20,           # Equal weight
        ccj_weight=0.10,              # Half weight
        asset_weight=0.10,            # Half weight
        # Total: 0.20 + 0.20 + 0.20 + 0.20 + 0.10 + 0.10 = 1.00 ✅
        
        high_dti_threshold=0.5        # Standard threshold
    )
    
    engagement_config = EngagementConfig(
        # ALL weights specified to sum to 1.0 (perfectly balanced)
        click_rate_weight= (1/3),      # Equal weight
        avg_time_weight= (1/3),        # Equal weight  
        interactions_weight=(1/3),    # Equal weight (rounding)
        # Total: 0.333 + 0.333 + 0.334 = 1.000 ✅
        
        high_engagement_threshold=0.5,   # Standard thresholds
        medium_engagement_threshold=0.25
    )
    
    return financial_config, engagement_config

# ================================
# USAGE EXAMPLES
# ================================

if __name__ == "__main__":
    print("🔧 REFACTORED FINANCIAL RECOMMENDER")
    print("=" * 60)
    
    try:
        # Standard configuration
        print("\n📊 RUNNING WITH STANDARD CONFIGURATION")
        print("Engagement thresholds: High=0.5, Medium=0.25")
        print("Financial thresholds: Excellent≥0.8, Good≥0.65, Fair≥0.45")
        standard_recommender = RefactoredFinancialRecommender('joined_user_table.csv')
        standard_features, standard_analysis = standard_recommender.run_analysis("standard")
        
        # Conservative configuration
        print("\n📊 RUNNING WITH CONSERVATIVE CONFIGURATION")
        print("Engagement thresholds: High=0.6, Medium=0.35 (STRICTER)")
        print("Financial thresholds: Excellent≥0.85, Good≥0.70 (STRICTER)")
        fin_config, eng_config = create_conservative_config()
        conservative_recommender = RefactoredFinancialRecommender(
            'joined_user_table.csv',
            financial_config=fin_config,
            engagement_config=eng_config
        )
        conservative_features, conservative_analysis = conservative_recommender.run_analysis("conservative")
        
        # Aggressive configuration
        print("\n📊 RUNNING WITH AGGRESSIVE CONFIGURATION")
        print("Engagement thresholds: High=0.4, Medium=0.15 (MORE LENIENT)")
        print("Financial thresholds: Excellent≥0.75, Good≥0.60 (MORE LENIENT)")
        fin_config_agg, eng_config_agg = create_aggressive_config()
        aggressive_recommender = RefactoredFinancialRecommender(
            'joined_user_table.csv',
            financial_config=fin_config_agg,
            engagement_config=eng_config_agg
        )
        aggressive_features, aggressive_analysis = aggressive_recommender.run_analysis("aggressive")
        
        # Compare segmentation results
        print("\n🔍 SEGMENTATION COMPARISON:")
        print("=" * 60)
        
        standard_segments = standard_features['enhanced_segment'].value_counts()
        conservative_segments = conservative_features['enhanced_segment'].value_counts()
        aggressive_segments = aggressive_features['enhanced_segment'].value_counts()
        
        print(f"\n{'Segment':<25} {'Standard':<10} {'Conservative':<12} {'Aggressive':<10}")
        print("-" * 65)
        
        all_segments = set(standard_segments.index) | set(conservative_segments.index) | set(aggressive_segments.index)
        for segment in sorted(all_segments):
            std_count = standard_segments.get(segment, 0)
            cons_count = conservative_segments.get(segment, 0)
            agg_count = aggressive_segments.get(segment, 0)
            print(f"{segment:<25} {std_count:<10} {cons_count:<12} {agg_count:<10}")
        
        print("\n💡 REFACTORING BENEFITS:")
        print("✅ NO MORE MAGIC NUMBERS - All thresholds configurable")
        print("✅ Configurable parameters for different business strategies")
        print("✅ Modular components with single responsibilities") 
        print("✅ Better testability and maintainability")
        print("✅ Separation of concerns across classes")
        print("✅ Type safety with dataclasses")
        print("✅ Strategy pattern for flexible segmentation")
        print("✅ Easy A/B testing of different configurations")
        
        print(f"\n🎯 CONFIGURATION IMPACT VISIBLE:")
        print(f"Conservative config creates more 'Activation_Needed' users (stricter thresholds)")
        print(f"Aggressive config creates more 'Premium'/'Growth' users (lenient thresholds)")
        print(f"Standard config provides balanced segmentation")
        
        print(f"\n📁 FILES GENERATED:")
        print("=" * 60)
        print("🔹 STANDARD CONFIGURATION:")
        print("   • standard_financial_dashboard.png")
        print("   • standard_correlation_heatmap.png") 
        print("   • standard_clustering_visualization.png")
        
        print("\n🔹 CONSERVATIVE CONFIGURATION:")
        print("   • conservative_financial_dashboard.png")
        print("   • conservative_correlation_heatmap.png")
        print("   • conservative_clustering_visualization.png")
        
        print("\n🔹 AGGRESSIVE CONFIGURATION:")
        print("   • aggressive_financial_dashboard.png")
        print("   • aggressive_correlation_heatmap.png") 
        print("   • aggressive_clustering_visualization.png")
        
        print(f"\n💡 TOTAL: 9 visualization files for comparative analysis!")
        print(f"Now you can compare how different business strategies affect user segmentation")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Run enhanced_financial_recommender.py first to generate the data") 