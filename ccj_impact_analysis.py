import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('joined_user_table.csv')

# Get unique users with their financial data
user_financial = df.groupby('user_id').first()[
    ['total_debt', 'credit_score', 'missed_payments', 
     'has_mortgage', 'has_car', 'has_ccj', 'dti_ratio', 'income']
]

print('🚨 CCJ IMPACT ANALYSIS 🚨')
print('=' * 60)

# Show CCJ distribution
ccj_counts = user_financial['has_ccj'].value_counts()
print(f'Users with CCJ: {ccj_counts.get(1, 0)}')
print(f'Users without CCJ: {ccj_counts.get(0, 0)}')
print(f'Total users: {len(user_financial)}')

print('\n📊 CCJ COMPONENT CALCULATION:')
print('=' * 40)
print('If user has CCJ (has_ccj = 1): CCJ component = 0.0')
print('If user has NO CCJ (has_ccj = 0): CCJ component = 1.0')
print('Weight in overall score: 10%')
print('💥 IMMEDIATE IMPACT: Having CCJ reduces financial health score by 10%')

# Calculate financial health scores with detailed breakdown
financial_scores = []
ccj_impacts = []
detailed_breakdown = []

for user_id, user in user_financial.iterrows():
    # All components (same as in enhanced_financial_recommender.py)
    credit_component = min(user['credit_score'] / 1000, 1.0)
    dti_component = max(0, 1 - user['dti_ratio'])
    missed_payments_component = max(0, 1 - (user['missed_payments'] / 10))
    income_component = min(user['income'] / 100000, 1.0)
    
    # 🎯 CCJ Component - THE KEY IMPACT
    ccj_component = 0.0 if user['has_ccj'] else 1.0
    
    asset_component = (user['has_mortgage'] * 0.6 + user['has_car'] * 0.4)
    
    # Calculate what score would be WITHOUT CCJ penalty
    score_without_ccj_penalty = (
        credit_component * 0.30 +
        dti_component * 0.25 + 
        missed_payments_component * 0.15 +
        income_component * 0.15 +
        1.0 * 0.10 +  # Assume no CCJ
        asset_component * 0.05
    )
    
    # Calculate actual score (with CCJ penalty if applicable)
    actual_score = (
        credit_component * 0.30 +
        dti_component * 0.25 +
        missed_payments_component * 0.15 +
        income_component * 0.15 +
        ccj_component * 0.10 +  # 🚨 CCJ penalty applied here
        asset_component * 0.05
    )
    
    ccj_impact = score_without_ccj_penalty - actual_score
    financial_scores.append(actual_score)
    ccj_impacts.append(ccj_impact)
    
    detailed_breakdown.append({
        'user_id': user_id,
        'credit_score': user['credit_score'],
        'has_ccj': user['has_ccj'],
        'ccj_component': ccj_component,
        'financial_health_score': actual_score,
        'ccj_penalty': ccj_impact,
        'would_be_without_ccj': score_without_ccj_penalty
    })

user_financial['financial_health_score'] = financial_scores
user_financial['ccj_penalty'] = ccj_impacts

print('\n💔 EXAMPLES OF CCJ DAMAGE:')
print('=' * 50)

# Show users with CCJ
ccj_users = user_financial[user_financial['has_ccj'] == 1].head(8)
print('USERS WITH CCJ (showing financial damage):')
for user_id, user in ccj_users.iterrows():
    print(f'  {user_id}: Credit {user["credit_score"]:3.0f} | Health Score {user["financial_health_score"]:.3f} | 💥 CCJ Penalty: -{user["ccj_penalty"]:.3f}')

print('\n✅ USERS WITHOUT CCJ (no penalty):')  
no_ccj_users = user_financial[user_financial['has_ccj'] == 0].head(8)
for user_id, user in no_ccj_users.iterrows():
    print(f'  {user_id}: Credit {user["credit_score"]:3.0f} | Health Score {user["financial_health_score"]:.3f} | ✅ CCJ Penalty: -{user["ccj_penalty"]:.3f}')

# Statistical comparison
print('\n📈 STATISTICAL IMPACT COMPARISON:')
print('=' * 50)
with_ccj = user_financial[user_financial['has_ccj'] == 1]['financial_health_score']
without_ccj = user_financial[user_financial['has_ccj'] == 0]['financial_health_score']

print(f'💔 Average financial health WITH CCJ:    {with_ccj.mean():.3f}')
print(f'✅ Average financial health WITHOUT CCJ: {without_ccj.mean():.3f}')
print(f'📉 Average difference due to CCJ:        {without_ccj.mean() - with_ccj.mean():.3f}')

print(f'\n🎯 KEY FINDINGS:')
print(f'  • CCJ penalty is exactly 10% of total score for ALL users with CCJ')
print(f'  • {ccj_counts.get(1, 0)} users are penalized by 0.100 points each')
print(f'  • CCJ users average {with_ccj.mean():.3f} vs {without_ccj.mean():.3f} for non-CCJ users')

# Show specific examples of "what if" scenarios
print(f'\n🔍 "WHAT IF" CCJ SCENARIOS:')
print('=' * 50)
print('If CCJ users had NO CCJ, their scores would be:')

ccj_examples = detailed_breakdown[:5]  # First 5 users
for example in ccj_examples:
    if example['has_ccj'] == 1:
        print(f"  {example['user_id']}: Current {example['financial_health_score']:.3f} → Would be {example['would_be_without_ccj']:.3f} (+{example['ccj_penalty']:.3f})")

print(f'\n🏷️  FINANCIAL CATEGORIZATION IMPACT:')
print('=' * 50)

# Show how CCJ affects financial categories
def categorize_financial_health(score):
    if score >= 0.7:
        return "Excellent"
    elif score >= 0.5:
        return "Good"  
    elif score >= 0.3:
        return "Fair"
    else:
        return "Poor"

ccj_categories = []
no_ccj_categories = []

for example in detailed_breakdown:
    current_cat = categorize_financial_health(example['financial_health_score'])
    would_be_cat = categorize_financial_health(example['would_be_without_ccj'])
    
    if example['has_ccj'] == 1:
        ccj_categories.append({
            'user_id': example['user_id'],
            'current_category': current_cat,
            'would_be_category': would_be_cat,
            'category_impact': 'DOWNGRADED' if current_cat != would_be_cat else 'SAME'
        })

print('Users whose financial category is hurt by CCJ:')
downgrades = [x for x in ccj_categories if x['category_impact'] == 'DOWNGRADED']
for downgrade in downgrades[:5]:
    print(f"  {downgrade['user_id']}: {downgrade['would_be_category']} → {downgrade['current_category']} (due to CCJ)")

print(f'\n📊 CCJ Category Impact Summary:')
print(f'  • {len(downgrades)} users downgraded to worse category due to CCJ')
print(f'  • {len(ccj_categories) - len(downgrades)} CCJ users remain in same category')
print(f'  • CCJ penalty of 0.100 can push users from "Good" to "Fair" category') 