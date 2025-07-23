from enhanced_financial_recommender import FinanciallyAwareRecommender

# Initialize the recommender
print("🔍 DTI ANALYSIS EXAMPLES")
print("=" * 60)

recommender = FinanciallyAwareRecommender('joined_user_table.csv')

# Run initial setup
recommender.create_enhanced_user_features()

print("\n📊 EXAMPLE 1: All users sorted by DTI (highest first)")
print("-" * 50)
all_users_by_dti = recommender.sort_users_by_dti()

print("\n📊 EXAMPLE 2: Only show High DTI users (≥50%)")
print("-" * 50) 
high_dti_users = recommender.sort_users_by_dti(min_dti=0.5)

print("\n📊 EXAMPLE 3: Top 5 worst DTI cases")
print("-" * 50)
worst_dti_users = recommender.sort_users_by_dti(top_n=5)

print("\n📊 EXAMPLE 4: Best DTI users (lowest first)")
print("-" * 50)
best_dti_users = recommender.sort_users_by_dti(ascending=True, top_n=10)

print("\n📊 EXAMPLE 5: Critical DTI cases (≥50%) - silent mode")
print("-" * 50)
critical_dti = recommender.sort_users_by_dti(min_dti=0.50, show_details=False)
if len(critical_dti) > 0:
    print(f"Found {len(critical_dti)} users with critical DTI ≥50%:")
    print(critical_dti[['dti_percentage', 'financial_category', 'credit_score', 'income', 'total_debt']].head())
else:
    print("No users found with critical DTI ≥50%")

print("\n✅ DTI Analysis Complete!")
print("💡 You can also access the returned DataFrames for further analysis:")
print("   - all_users_by_dti: Full dataset sorted by DTI")
print("   - high_dti_users: Only users with DTI ≥50%") 
print("   - worst_dti_users: Top 5 highest DTI users")
print("   - best_dti_users: Top 10 lowest DTI users")
print("   - critical_dti: Users with DTI ≥50%") 