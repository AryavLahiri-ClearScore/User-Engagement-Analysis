import pandas as pd
import numpy as np

def create_realistic_financial_health_score(user_financial):
    """Create financial health score with realistic UK-based categorization"""
    
    financial_scores = []
    
    for _, user in user_financial.iterrows():
        # Credit Score Component (based on UK Experian scale 0-999)
        credit_score = user['credit_score']
        if credit_score >= 881:
            credit_component = 1.0      # Excellent (881-999)
        elif credit_score >= 721:
            credit_component = 0.75     # Good (721-880)
        elif credit_score >= 561:
            credit_component = 0.5      # Fair (561-720)
        else:
            credit_component = 0.25     # Poor (0-560)
        
        # DTI Ratio Component (UK standards)
        dti_ratio = user['dti_ratio']
        if dti_ratio <= 0.3:
            dti_component = 1.0         # Excellent (<30%)
        elif dti_ratio <= 0.4:
            dti_component = 0.75        # Good (30-40%)
        elif dti_ratio <= 0.5:
            dti_component = 0.5         # Fair (40-50%)
        else:
            dti_component = 0.25        # Poor (>50%)
        
        # Income Component (UK median ~£31,400)
        income = user['income']
        if income >= 50000:
            income_component = 1.0      # High income
        elif income >= 31400:
            income_component = 0.75     # Above median
        elif income >= 20000:
            income_component = 0.5      # Moderate income
        else:
            income_component = 0.25     # Low income
        
        # Missed Payments Component
        missed_payments = user['missed_payments']
        if missed_payments == 0:
            payment_component = 1.0     # Perfect payment history
        elif missed_payments <= 2:
            payment_component = 0.75    # Minor issues
        elif missed_payments <= 4:
            payment_component = 0.5     # Moderate issues
        else:
            payment_component = 0.25    # Serious issues
        
        # CCJ Component (binary impact)
        ccj_component = 0.0 if user['has_ccj'] else 1.0
        
        # Asset Component 
        asset_component = (user['has_mortgage'] * 0.6 + user['has_car'] * 0.4)
        
        # Weighted composite score
        financial_health = (
            credit_component * 0.35 +       # Credit score most important
            dti_component * 0.25 +          # Debt management crucial
            payment_component * 0.20 +      # Payment history important
            income_component * 0.10 +       # Income level
            ccj_component * 0.05 +          # Legal issues (minor weight but critical)
            asset_component * 0.05          # Asset ownership (least important)
        )
        
        financial_scores.append(financial_health)
    
    user_financial['financial_health_score'] = financial_scores
    
    # ABSOLUTE THRESHOLDS based on real financial health standards
    def categorize_financial_health_realistic(score):
        if score >= 0.8:
            return "Excellent"      # Strong across all metrics
        elif score >= 0.65:
            return "Good"           # Above average financial health
        elif score >= 0.45:
            return "Fair"           # Some concerns but manageable
        else:
            return "Poor"           # Significant financial challenges
    
    user_financial['financial_category'] = user_financial['financial_health_score'].apply(
        categorize_financial_health_realistic
    )
    
    print("Realistic Financial Health Distribution:")
    print(user_financial['financial_category'].value_counts())
    print(f"Score ranges: {user_financial['financial_health_score'].min():.3f} - {user_financial['financial_health_score'].max():.3f}")
    
    return user_financial

# Example of what realistic distribution might look like:
# Poor: 15-25% (users with serious financial challenges)
# Fair: 35-45% (users with some financial concerns)  
# Good: 25-35% (users with decent financial health)
# Excellent: 5-15% (users with strong financial position)

print("Key improvements:")
print("✅ Credit scores use actual UK Experian ranges")
print("✅ DTI ratios use recognized lending standards")
print("✅ Income thresholds based on UK median income")
print("✅ Categories reflect absolute financial health, not relative ranking")
print("✅ Distribution reflects real-world financial health patterns") 