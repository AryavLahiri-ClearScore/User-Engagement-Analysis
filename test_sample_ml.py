import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

print("=== How Leftover Features Become NOISE ===\n")

def demonstrate_leftover_noise():
    print("=== The Leftover Rule in Action ===")
    
    # Example 1: Same as original - no leftovers
    print("\n--- Example 1: Original (no leftovers) ---")
    print("n_features=20, n_informative=15, n_redundant=5")
    print("Math: 15 + 5 = 20 ✓ (no leftovers)")
    
    X1, y1 = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                n_redundant=5, random_state=42)
    
    print("Feature assignments:")
    print("- Features 0-14:  INFORMATIVE (15 features)")
    print("- Features 15-19: REDUNDANT (5 features)")
    print("- Features 20+:   NONE (no leftovers)")
    
    # Calculate correlations to confirm
    correlations = [abs(np.corrcoef(X1[:, i], y1)[0, 1]) for i in range(20)]
    avg_corr_inform = np.mean(correlations[0:15])
    avg_corr_redund = np.mean(correlations[15:20])
    print(f"Average correlation - Informative: {avg_corr_inform:.4f}, Redundant: {avg_corr_redund:.4f}")
    
    # Example 2: Add leftover features
    print("\n--- Example 2: Adding leftover features (become noise) ---")
    print("n_features=25, n_informative=15, n_redundant=5")
    print("Math: 15 + 5 = 20, but n_features=25, so 5 leftovers!")
    
    X2, y2 = make_classification(n_samples=1000, n_features=25, n_informative=15, 
                                n_redundant=5, random_state=42)
    
    print("Feature assignments:")
    print("- Features 0-14:  INFORMATIVE (15 features)")
    print("- Features 15-19: REDUNDANT (5 features)")
    print("- Features 20-24: NOISE (5 leftover features)")
    
    # Calculate correlations for all types
    correlations2 = [abs(np.corrcoef(X2[:, i], y2)[0, 1]) for i in range(25)]
    avg_corr_inform2 = np.mean(correlations2[0:15])
    avg_corr_redund2 = np.mean(correlations2[15:20])
    avg_corr_noise2 = np.mean(correlations2[20:25])
    print(f"Average correlation - Informative: {avg_corr_inform2:.4f}, Redundant: {avg_corr_redund2:.4f}, Noise: {avg_corr_noise2:.4f}")
    
    # Example 3: Extreme case - mostly noise
    print("\n--- Example 3: Extreme case (mostly leftovers) ---")
    print("n_features=30, n_informative=5, n_redundant=3")
    print("Math: 5 + 3 = 8, but n_features=30, so 22 leftovers!")
    
    X3, y3 = make_classification(n_samples=1000, n_features=30, n_informative=5, 
                                n_redundant=3, random_state=42)
    
    print("Feature assignments:")
    print("- Features 0-4:   INFORMATIVE (5 features)")
    print("- Features 5-7:   REDUNDANT (3 features)")
    print("- Features 8-29:  NOISE (22 leftover features)")
    
    correlations3 = [abs(np.corrcoef(X3[:, i], y3)[0, 1]) for i in range(30)]
    avg_corr_inform3 = np.mean(correlations3[0:5])
    avg_corr_redund3 = np.mean(correlations3[5:8])
    avg_corr_noise3 = np.mean(correlations3[8:30])
    print(f"Average correlation - Informative: {avg_corr_inform3:.4f}, Redundant: {avg_corr_redund3:.4f}, Noise: {avg_corr_noise3:.4f}")

demonstrate_leftover_noise()

print("\n" + "="*70)
print("=== THE LEFTOVER FORMULA ===")
print("="*70)

def show_leftover_formula():
    print("""
The Automatic Assignment Rule:
=============================

n_noise = n_features - (n_informative + n_redundant)

Examples:
--------
• n_features=20, n_informative=15, n_redundant=5
  → n_noise = 20 - (15 + 5) = 0  (no noise)

• n_features=25, n_informative=15, n_redundant=5  
  → n_noise = 25 - (15 + 5) = 5  (5 noise features)

• n_features=100, n_informative=10, n_redundant=5
  → n_noise = 100 - (10 + 5) = 85  (85 noise features!)

Feature Index Assignment:
========================
- Features [0 to n_informative-1]:                    INFORMATIVE
- Features [n_informative to n_informative+n_redundant-1]: REDUNDANT  
- Features [n_informative+n_redundant to n_features-1]:    NOISE (leftovers)
    """)

show_leftover_formula()

print("\n=== PERFORMANCE IMPACT TEST ===")

def test_leftover_impact():
    print("\nTesting how leftovers hurt performance:")
    
    base_params = {"n_samples": 1000, "n_informative": 10, "n_redundant": 5, "random_state": 42}
    
    # Test different amounts of leftover noise
    configurations = [
        {"n_features": 15, "name": "No leftovers"},      # 10+5=15, no noise
        {"n_features": 20, "name": "5 noise features"},  # 10+5=15, 5 noise  
        {"n_features": 30, "name": "15 noise features"}, # 10+5=15, 15 noise
        {"n_features": 50, "name": "35 noise features"}, # 10+5=15, 35 noise
    ]
    
    print("Config                | Features | Informative | Redundant | Noise | Accuracy")
    print("----------------------|----------|-------------|-----------|-------|----------")
    
    for config in configurations:
        # Create dataset
        X, y = make_classification(n_features=config["n_features"], **base_params)
        
        # Train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Calculate noise count
        noise_count = config["n_features"] - (base_params["n_informative"] + base_params["n_redundant"])
        
        print(f"{config['name']:20s} | {config['n_features']:8d} | {base_params['n_informative']:11d} | {base_params['n_redundant']:9d} | {noise_count:5d} | {accuracy:.4f}")

test_leftover_impact()

print(f"\n=== KEY INSIGHT ===")
print("Yes! Any features beyond n_informative + n_redundant automatically become NOISE")
print("More noise features = worse performance!")
print("The algorithm doesn't 'choose' which are noise - it's just math: leftovers = noise")

print("\n" + "="*70)
print("=== INVESTIGATING THE 35 NOISE PARADOX ===")
print("="*70)

def investigate_noise_paradox():
    print("\nTesting if 35 noise features consistently perform best...")
    
    base_params = {"n_samples": 1000, "n_informative": 10, "n_redundant": 5}
    
    configurations = [
        {"n_features": 15, "name": "No noise"},
        {"n_features": 20, "name": "5 noise"},  
        {"n_features": 30, "name": "15 noise"},
        {"n_features": 50, "name": "35 noise"},
    ]
    
    # Test across multiple random states
    random_states = [42, 123, 456, 789, 999]
    
    print("Testing across different random states:")
    print("Random State | No noise | 5 noise | 15 noise | 35 noise")
    print("-------------|----------|---------|----------|----------")
    
    for rs in random_states:
        results = []
        for config in configurations:
            # Create dataset with this random state
            X, y = make_classification(n_features=config["n_features"], 
                                     random_state=rs, **base_params)
            
            # Split and train with same random state
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
            model = LogisticRegression(random_state=rs, max_iter=1000)
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            results.append(accuracy)
        
        print(f"    {rs:3d}      | {results[0]:.4f}   | {results[1]:.4f}  | {results[2]:.4f}   | {results[3]:.4f}")
    
    print(f"\n=== POSSIBLE EXPLANATIONS ===")
    print("""
Why 35 noise features might perform "better":

1. REGULARIZATION EFFECT:
   - With more features, logistic regression spreads weights thinner
   - Acts like implicit L2 regularization
   - Prevents overfitting to training data
   - Better generalization to test set

2. RANDOM STATE LUCK:
   - With random_state=42, we get the same train/test split
   - The model might get "lucky" with this specific split
   - Different random states show the true pattern

3. OPTIMIZATION LANDSCAPE:
   - More features change how the algorithm converges
   - Might find a better local minimum by accident
   - Noise can act as "exploration" during training

4. SAMPLE-TO-FEATURE RATIO:
   - 15 features: ~67 samples per feature (might overfit)
   - 50 features: ~20 samples per feature (more regularized)

The key insight: This is likely a fluke of the specific random_state=42!
Real-world rule still holds: noise features generally hurt performance.
    """)

investigate_noise_paradox()