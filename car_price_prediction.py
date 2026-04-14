"""
Car Price Prediction with Machine Learning
Task 3: Code Alpha Internship

This script performs:
1. Data loading and exploration
2. Feature engineering and preprocessing
3. Model training (Linear Regression & Random Forest)
4. Model evaluation with metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("="*60)
print("CAR PRICE PREDICTION - MACHINE LEARNING PROJECT")
print("="*60)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1] Loading Dataset...")
df = pd.read_csv('car_data.csv')
print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n[2] Exploratory Data Analysis")
print("-" * 60)

print("\n📊 Dataset Info:")
print(df.info())

print("\n📊 First 5 Rows:")
print(df.head())

print("\n📊 Statistical Summary:")
print(df.describe())

print("\n📊 Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found!")
else:
    print(missing[missing > 0])

print("\n📊 Duplicate Rows:")
duplicates = df.duplicated().sum()
print(f"  Total duplicates: {duplicates}")

print("\n📊 Target Variable (Selling_Price) Distribution:")
print(f"  Mean: {df['Selling_Price'].mean():.2f}")
print(f"  Median: {df['Selling_Price'].median():.2f}")
print(f"  Std Dev: {df['Selling_Price'].std():.2f}")
print(f"  Min: {df['Selling_Price'].min():.2f}")
print(f"  Max: {df['Selling_Price'].max():.2f}")

# Visualize target distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['Selling_Price'], bins=30, edgecolor='black', color='skyblue')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.title('Distribution of Selling Price')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['Selling_Price'], vert=True)
plt.ylabel('Selling Price')
plt.title('Boxplot of Selling Price')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('selling_price_distribution.png', dpi=300, bbox_inches='tight')
print("\n  ✓ Saved: selling_price_distribution.png")
plt.close()

# ============================================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# ============================================================================
print("\n[3] Feature Engineering & Preprocessing")
print("-" * 60)

# Create a copy for preprocessing
df_processed = df.copy()

# Create Age feature from Year
current_year = 2024
df_processed['Age'] = current_year - df_processed['Year']
print(f"  ✓ Created 'Age' feature (Current Year: {current_year})")

# Drop unnecessary columns
df_processed = df_processed.drop(['Car_Name', 'Year'], axis=1)
print("  ✓ Dropped 'Car_Name' and 'Year' columns")

print("\n📊 Categorical Features:")
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    print(f"  - {col}: {df_processed[col].nunique()} unique values")
    print(f"    Values: {df_processed[col].unique()}")

# One-Hot Encoding for categorical variables
df_encoded = pd.get_dummies(df_processed, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
print(f"\n  ✓ Applied One-Hot Encoding")
print(f"  Final feature count: {df_encoded.shape[1] - 1} features")

# Correlation matrix
plt.figure(figsize=(14, 10))
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: correlation_matrix.png")
plt.close()

# Show top correlations with Selling_Price
print("\n📊 Top Features Correlated with Selling_Price:")
correlations = correlation_matrix['Selling_Price'].sort_values(ascending=False)
print(correlations.head(10))

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n[4] Splitting Data into Train and Test Sets")
print("-" * 60)

X = df_encoded.drop('Selling_Price', axis=1)
y = df_encoded['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")
print(f"  Features: {X_train.shape[1]}")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n[5] Training Machine Learning Models")
print("-" * 60)

# Linear Regression
print("\n🤖 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("  ✓ Linear Regression trained!")

# Random Forest Regressor
print("\n🤖 Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("  ✓ Random Forest trained!")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n[6] Model Evaluation")
print("=" * 60)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Linear Regression Metrics
print("\n📈 LINEAR REGRESSION PERFORMANCE:")
print("-" * 60)
lr_r2 = r2_score(y_test, y_pred_lr)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)

print(f"  R² Score:              {lr_r2:.4f}")
print(f"  Mean Absolute Error:   {lr_mae:.4f}")
print(f"  Mean Squared Error:    {lr_mse:.4f}")
print(f"  Root Mean Squared Error: {lr_rmse:.4f}")

# Random Forest Metrics
print("\n📈 RANDOM FOREST PERFORMANCE:")
print("-" * 60)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)

print(f"  R² Score:              {rf_r2:.4f}")
print(f"  Mean Absolute Error:   {rf_mae:.4f}")
print(f"  Mean Squared Error:    {rf_mse:.4f}")
print(f"  Root Mean Squared Error: {rf_rmse:.4f}")

# Model Comparison
print("\n📊 MODEL COMPARISON:")
print("-" * 60)
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R² Score': [lr_r2, rf_r2],
    'MAE': [lr_mae, rf_mae],
    'RMSE': [lr_rmse, rf_rmse]
})
print(comparison_df.to_string(index=False))

best_model = 'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'
print(f"\n🏆 Best Model: {best_model}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n[7] Generating Visualizations")
print("-" * 60)

# Actual vs Predicted - Linear Regression
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.6, edgecolors='k', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title(f'Linear Regression\nR² = {lr_r2:.4f}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# Actual vs Predicted - Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.6, edgecolors='k', s=50, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title(f'Random Forest\nR² = {rf_r2:.4f}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: actual_vs_predicted.png")
plt.close()

# Feature Importance (Random Forest)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10), color='teal')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: feature_importance.png")
plt.close()

print("\n📊 Top 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Residual Plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
residuals_lr = y_test - y_pred_lr
plt.scatter(y_pred_lr, residuals_lr, alpha=0.6, edgecolors='k', s=50)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot - Linear Regression', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
residuals_rf = y_test - y_pred_rf
plt.scatter(y_pred_rf, residuals_rf, alpha=0.6, edgecolors='k', s=50, color='green')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot - Random Forest', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: residual_plots.png")
plt.close()

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Dataset: {df.shape[0]} cars with {df.shape[1]} original features")
print(f"✓ Features after preprocessing: {X.shape[1]}")
print(f"✓ Best Model: {best_model}")
print(f"✓ Best R² Score: {max(lr_r2, rf_r2):.4f}")
print(f"✓ Visualizations saved: 4 PNG files")
print("\n🎯 Real-world Application:")
print("   This model can help:")
print("   - Car dealerships price their inventory")
print("   - Buyers estimate fair market value")
print("   - Sellers determine competitive pricing")
print("   - Insurance companies assess vehicle worth")
print("=" * 60)
print("✅ Car Price Prediction Project Completed Successfully!")
print("=" * 60)
