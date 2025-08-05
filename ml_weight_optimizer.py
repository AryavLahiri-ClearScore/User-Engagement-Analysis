import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class MLWeightOptimizer:
    """
    Use machine learning to optimize engagement score weights
    """
    
    def __init__(self, user_features_df):
        self.df = user_features_df
        self.optimal_weights = None
        self.model = None
        
    def method_1_supervised_learning(self, target_variable='financial_health_score'):
        """
        Method 1: Use supervised learning with a business outcome target
        
        Target could be:
        - financial_health_score (engagement should predict financial wellness)
        - retention_rate (if you have retention data)
        - conversion_rate (if you have conversion data)
        - revenue_per_user (if you have revenue data)
        """
        print("🤖 METHOD 1: Supervised Learning Weight Optimization")
        print("=" * 60)
        
        # Features for engagement (same as current manual formula)
        engagement_features = ['click_rate', 'avg_time_viewed', 'total_interactions']
        
        # Check if all required columns exist
        missing_features = [col for col in engagement_features if col not in self.df.columns]
        if missing_features:
            print(f"❌ Missing engagement features: {missing_features}")
            print("Available columns:", list(self.df.columns))
            return None
            
        if target_variable not in self.df.columns:
            print(f"❌ Missing target variable: {target_variable}")
            print("Available columns:", list(self.df.columns))
            return None
        
        # Prepare data
        X = self.df[engagement_features].fillna(0)
        y = self.df[target_variable].fillna(0)
        
        # Normalize features (important for weight interpretation)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train linear regression (weights are directly interpretable)
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Extract learned weights
        raw_weights = lr_model.coef_
        
        # Normalize weights to sum to 1 (like current system)
        normalized_weights = raw_weights / np.sum(np.abs(raw_weights))
        
        print(f"Target variable: {target_variable}")
        print(f"Current manual weights: [0.4, 0.3, 0.3]")
        print(f"ML-learned weights:     {normalized_weights}")
        
        # Compare performance
        y_pred = lr_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Model R² score: {r2:.3f}")
        
        # Create optimized engagement score
        self.optimal_weights = dict(zip(engagement_features, normalized_weights))
        
        return self.optimal_weights
    
    def method_2_pca_weights(self):
        """
        Method 2: Use PCA to find optimal linear combination
        """
        print("\n🤖 METHOD 2: PCA-Based Weight Optimization")
        print("=" * 60)
        
        from sklearn.decomposition import PCA
        
        engagement_features = ['click_rate', 'avg_time_viewed', 'total_interactions']
        
        # Check if all required columns exist
        missing_features = [col for col in engagement_features if col not in self.df.columns]
        if missing_features:
            print(f"❌ Missing engagement features: {missing_features}")
            return None
            
        X = self.df[engagement_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=1)  # Get the first principal component
        pca.fit(X_scaled)
        
        # The first principal component gives us optimal weights
        pc1_weights = pca.components_[0]
        
        # Normalize to positive weights summing to 1
        normalized_weights = np.abs(pc1_weights) / np.sum(np.abs(pc1_weights))
        
        print(f"Current manual weights: [0.4, 0.3, 0.3]")
        print(f"PCA-optimal weights:    {normalized_weights}")
        print(f"Explained variance:     {pca.explained_variance_ratio_[0]:.1%}")
        
        return dict(zip(engagement_features, normalized_weights))
    
    def method_3_genetic_algorithm(self, target_variable='financial_health_score', 
                                  population_size=50, generations=100):
        """
        Method 3: Genetic Algorithm to evolve optimal weights
        """
        print("\n🤖 METHOD 3: Genetic Algorithm Optimization")
        print("=" * 60)
        
        def calculate_fitness(weights, features, target):
            """Calculate how good a set of weights is"""
            # Create engagement score with these weights
            engagement_scores = (
                features['click_rate'] * weights[0] +
                features['avg_time_viewed'] * weights[1] +
                features['total_interactions'] * weights[2]
            )
            
            # Correlation with target variable (higher = better)
            correlation = np.corrcoef(engagement_scores, target)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        
        def mutate(weights, mutation_rate=0.1):
            """Randomly modify weights"""
            mutated = weights.copy()
            for i in range(len(mutated)):
                if np.random.random() < mutation_rate:
                    mutated[i] += np.random.normal(0, 0.05)
                    mutated[i] = max(0, mutated[i])  # Keep positive
            
            # Normalize to sum to 1
            mutated = mutated / np.sum(mutated)
            return mutated
        
        def crossover(parent1, parent2):
            """Combine two parent weight sets"""
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            return child / np.sum(child)  # Normalize
        
        # Prepare data
        engagement_features = ['click_rate', 'avg_time_viewed', 'total_interactions']
        
        # Check if all required columns exist
        missing_features = [col for col in engagement_features if col not in self.df.columns]
        if missing_features:
            print(f"❌ Missing engagement features: {missing_features}")
            return None
            
        if target_variable not in self.df.columns:
            print(f"❌ Missing target variable: {target_variable}")
            return None
        
        features = self.df[engagement_features].fillna(0)
        target = self.df[target_variable].fillna(0)
        
        # Initialize random population
        population = []
        for _ in range(population_size):
            weights = np.random.random(3)
            weights = weights / np.sum(weights)  # Normalize
            population.append(weights)
        
        best_fitness_history = []
        
        # Evolution loop
        for generation in range(generations):
            # Calculate fitness for each individual
            fitness_scores = [
                calculate_fitness(weights, features, target) 
                for weights in population
            ]
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # Selection: keep top 50%
            sorted_indices = np.argsort(fitness_scores)[::-1]
            survivors = [population[i] for i in sorted_indices[:population_size//2]]
            
            # Create new generation
            new_population = survivors.copy()
            
            # Fill rest with crossover and mutation
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(len(survivors), 2, replace=False)
                child = crossover(survivors[parent1], survivors[parent2])
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Get best solution
        final_fitness = [calculate_fitness(w, features, target) for w in population]
        best_idx = np.argmax(final_fitness)
        optimal_weights = population[best_idx]
        
        print(f"Current manual weights: [0.4, 0.3, 0.3]")
        print(f"GA-optimal weights:     {optimal_weights}")
        print(f"Best fitness:           {max(final_fitness):.3f}")
        
        return dict(zip(engagement_features, optimal_weights))
    
    def method_4_multi_objective_optimization(self):
        """
        Method 4: Optimize for multiple objectives simultaneously
        """
        print("\n🤖 METHOD 4: Multi-Objective Optimization")
        print("=" * 60)
        
        from scipy.optimize import differential_evolution
        
        # Check if all required columns exist
        engagement_features = ['click_rate', 'avg_time_viewed', 'total_interactions']
        required_columns = engagement_features + ['financial_health_score', 'credit_score', 'dti_ratio']
        
        missing_features = [col for col in required_columns if col not in self.df.columns]
        if missing_features:
            print(f"❌ Missing required features: {missing_features}")
            return None
        
        def multi_objective_fitness(weights):
            """Optimize for multiple business objectives"""
            weights = weights / np.sum(weights)  # Normalize
            
            engagement_features = ['click_rate', 'avg_time_viewed', 'total_interactions']
            features = self.df[engagement_features].fillna(0)
            
            # Calculate engagement score with these weights
            engagement_scores = (
                features['click_rate'] * weights[0] +
                features['avg_time_viewed'] * weights[1] +
                features['total_interactions'] * weights[2]
            )
            
            # Multiple objectives to optimize
            obj1 = np.corrcoef(engagement_scores, self.df['financial_health_score'])[0, 1]
            obj2 = np.corrcoef(engagement_scores, self.df['credit_score'])[0, 1] 
            obj3 = -np.corrcoef(engagement_scores, self.df['dti_ratio'])[0, 1]  # Negative because lower DTI is better
            
            # Combined objective (weighted sum)
            combined_objective = (
                obj1 * 0.4 +  # 40% weight on financial health correlation
                obj2 * 0.3 +  # 30% weight on credit score correlation  
                obj3 * 0.3    # 30% weight on DTI correlation (inverted)
            )
            
            return -combined_objective  # Minimize (so negate)
        
        # Optimize using differential evolution
        bounds = [(0.01, 0.98)] * 3  # Each weight between 1% and 98%
        result = differential_evolution(
            multi_objective_fitness, 
            bounds, 
            seed=42,
            maxiter=100
        )
        
        optimal_weights = result.x / np.sum(result.x)  # Normalize
        
        print(f"Current manual weights: [0.4, 0.3, 0.3]")
        print(f"Multi-obj weights:      {optimal_weights}")
        print(f"Optimization success:   {result.success}")
        
        engagement_features = ['click_rate', 'avg_time_viewed', 'total_interactions']
        return dict(zip(engagement_features, optimal_weights))
    
    def compare_all_methods(self):
        """Compare all optimization methods"""
        print("\n📊 COMPARISON OF ALL METHODS")
        print("=" * 70)
        
        # Current manual weights
        manual_weights = {'click_rate': 0.4, 'avg_time_viewed': 0.3, 'total_interactions': 0.3}
        
        # Run all methods with error handling
        methods_results = {}
        
        try:
            methods_results['Supervised'] = self.method_1_supervised_learning()
        except Exception as e:
            print(f"⚠️  Supervised learning failed: {e}")
            methods_results['Supervised'] = None
            
        try:
            methods_results['PCA'] = self.method_2_pca_weights()
        except Exception as e:
            print(f"⚠️  PCA method failed: {e}")
            methods_results['PCA'] = None
            
        try:
            methods_results['Genetic Alg'] = self.method_3_genetic_algorithm()
        except Exception as e:
            print(f"⚠️  Genetic algorithm failed: {e}")
            methods_results['Genetic Alg'] = None
            
        try:
            methods_results['Multi-Objective'] = self.method_4_multi_objective_optimization()
        except Exception as e:
            print(f"⚠️  Multi-objective optimization failed: {e}")
            methods_results['Multi-Objective'] = None
        
        # Create comparison table
        print(f"\n{'Method':<15} {'Click Rate':<12} {'Avg Time':<12} {'Interactions':<12} {'Status':<10}")
        print("-" * 70)
        
        # Manual weights (always available)
        print(f"{'Manual':<15} {manual_weights['click_rate']:<12.3f} {manual_weights['avg_time_viewed']:<12.3f} {manual_weights['total_interactions']:<12.3f} {'✅ OK':<10}")
        
        # ML methods
        for method_name, weights in methods_results.items():
            if weights is not None:
                print(f"{method_name:<15} {weights['click_rate']:<12.3f} {weights['avg_time_viewed']:<12.3f} {weights['total_interactions']:<12.3f} {'✅ OK':<10}")
            else:
                print(f"{method_name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'❌ FAILED':<10}")
        
        # Return successful results
        successful_weights = [manual_weights]
        for weights in methods_results.values():
            if weights is not None:
                successful_weights.append(weights)
        
        return successful_weights


if __name__ == "__main__":
    # Example usage
    print("🔬 ML WEIGHT OPTIMIZATION EXAMPLE")
    print("=" * 50)
    
    # Load your data (replace with actual path)
    try:
        df = pd.read_csv('joined_user_table.csv')
        
        # Check if we need to create enhanced features first
        if 'financial_health_score' not in df.columns:
            print("⚠️  Financial features not found. Creating them first...")
            print("This requires running the FinanciallyAwareRecommender first.")
            
            # Import and create the recommender to generate features
            from enhanced_financial_recommender import FinanciallyAwareRecommender
            
            recommender = FinanciallyAwareRecommender('joined_user_table.csv')
            enhanced_features = recommender.create_enhanced_user_features()
            
            print("✅ Enhanced features created!")
            df = enhanced_features
        
        # Create synthetic engagement features for demo if needed
        if 'click_rate' not in df.columns:
            print("Creating synthetic engagement features for demo...")
            np.random.seed(42)
            n_users = len(df)
            df['click_rate'] = np.random.beta(2, 3, n_users)  # Skewed toward lower values
            df['avg_time_viewed'] = np.random.gamma(2, 30, n_users)  # Avg around 60 seconds
            df['total_interactions'] = np.random.poisson(8, n_users)  # Avg around 8 interactions
        
        # Initialize optimizer
        optimizer = MLWeightOptimizer(df)
        
        # Compare all methods
        weight_results = optimizer.compare_all_methods()
        
        print("\n💡 RECOMMENDATIONS:")
        print("=" * 30)
        print("1. If you have business outcome data (revenue, retention), use Supervised Learning")
        print("2. If you want to maximize data variance, use PCA weights")
        print("3. If you have multiple objectives, use Multi-Objective Optimization")  
        print("4. Consider A/B testing different weight combinations in production")
        
    except FileNotFoundError:
        print("❌ Error: joined_user_table.csv not found")
        print("Run the main enhanced_financial_recommender.py first to generate the data")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure enhanced_financial_recommender.py is in the same directory") 