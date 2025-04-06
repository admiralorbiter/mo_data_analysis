"""
Missouri Data Analysis - Step 6: Feature Engineering and Machine Learning Models
This script analyzes demographic and socioeconomic data from Missouri ZCTAs,
creates engineered features, and builds supervised machine learning models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
import os
from scipy import stats
warnings.filterwarnings('ignore')

# Create output directory
output_dir = os.path.join('analysis_output', 'ml_analysis')
os.makedirs(output_dir, exist_ok=True)

# Load the data
print("Loading data...")
df = pd.read_csv("mo.csv")

# Data overview
print("\nData Overview:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Feature engineering
print("\nPerforming feature engineering...")

# Handle missing values
# Replace empty strings with NaN
df = df.replace("", np.nan)

# For columns that should be numeric, convert to float
# First, get all columns after 'Male'
numeric_cols = df.columns[df.columns.get_loc('Male'):].tolist()

# Remove non-numeric columns (safer approach)
non_numeric = ['Geography']  # Add any other non-numeric columns here
numeric_cols = [col for col in numeric_cols if col not in non_numeric]

# Convert to numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate percentage features
print("Creating ratio/percentage features...")
# Gender ratio
df['Percent_Male'] = df['Male'] / (df['Male'] + df['Female']) * 100
df['Percent_Female'] = df['Female'] / (df['Male'] + df['Female']) * 100

# Age groups percentages
df['Total_Population'] = df['Male'] + df['Female']
df['Percent_Under_18'] = (df['Under 5 years'] + df['5 to 9 years'] + df['10 to 14 years'] + df['15 to 19 years'] * 0.8) / df['Total_Population'] * 100
df['Percent_18_to_64'] = (df['15 to 19 years'] * 0.2 + df['20 to 24 years'] + df['25 to 34 years'] + df['35 to 44 years'] + df['45 to 54 years'] + df['55 to 59 years'] + df['60 to 64 years']) / df['Total_Population'] * 100
df['Percent_65_plus'] = (df['65 to 74 years'] + df['75 to 84 years'] + df['85 years and over']) / df['Total_Population'] * 100

# Housing density
df['Housing_Density'] = df['Households Total'] / df['Total housing units']

# Education premium (earnings ratio of college vs high school)
df['Education_Premium'] = df["Total - MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings Bachelor's degree"] / df['Tota - MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings!!High school graduate (includes equivalency)']

# Foreign born percentage
df['Percent_Foreign_Born'] = df['Workers Foreign born'] / 100

# Language diversity
df['Percent_Non_English'] = df['Speak language other than English'] / 100

# Work from home percentage
df['Percent_Work_From_Home'] = (df['Total Workers 16 years and over'] - df['Workers 16 years and over who did not work from home']) / df['Total Workers 16 years and over'] * 100

# Income features
df['Income_Per_Capita'] = df['Households Mean income (dollars)'] / df['Households Total']

# Impute missing values for engineered features
print("Imputing missing values...")
imputer = SimpleImputer(strategy='median')
engineered_features = [
    'Percent_Male', 'Percent_Female', 'Percent_Under_18', 'Percent_18_to_64',
    'Percent_65_plus', 'Housing_Density', 'Education_Premium', 
    'Percent_Foreign_Born', 'Percent_Non_English', 'Percent_Work_From_Home',
    'Income_Per_Capita'
]

df[engineered_features] = imputer.fit_transform(df[engineered_features])

# Feature selection for modeling
print("\nPreparing features for modeling...")
features = [
    'Median age (years)', 'Percent_Male', 'Percent_Under_18', 'Percent_18_to_64',
    'Percent_65_plus', 'Housing_Density', 'Education_Premium', 
    'Percent_Foreign_Born', 'Percent_Non_English', 'Percent_Work_From_Home',
    'Workers Median age (years)', 'Income_Per_Capita'
]

# Feature correlation analysis
print("\nAnalyzing feature correlations...")
correlation_matrix = df[features + ['Percent with a disability', 'Households Mean income (dollars)']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
plt.close()

# Define target variables for regression tasks
print("\nSetting up regression models...")
# 1. Predicting mean household income
X = df[features].copy()
y_income = df['Households Mean income (dollars)']

# 2. Predicting disability rate
y_disability = df['Percent with a disability']

# Drop rows with NaN in features or target
mask = ~X.isna().any(axis=1) & ~y_income.isna() & ~y_disability.isna()
X = X[mask]
y_income = y_income[mask]
y_disability = y_disability[mask]

# Splitting the data
X_train_income, X_test_income, y_train_income, y_test_income = train_test_split(
    X, y_income, test_size=0.2, random_state=42
)

X_train_disability, X_test_disability, y_train_disability, y_test_disability = train_test_split(
    X, y_disability, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_income_scaled = scaler.fit_transform(X_train_income)
X_test_income_scaled = scaler.transform(X_test_income)

scaler_disability = StandardScaler()
X_train_disability_scaled = scaler_disability.fit_transform(X_train_disability)
X_test_disability_scaled = scaler_disability.transform(X_test_disability)

# Helper function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-squared: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Function to plot feature importances for tree-based models
def plot_feature_importance(model, features_list, title, filename):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(len(features_list)), importances[indices], align='center')
        plt.xticks(range(len(features_list)), [features_list[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

#===============================================
# MODEL 1: Predicting Household Mean Income
#===============================================
print("\n" + "="*50)
print("MODEL 1: Predicting Household Mean Income")
print("="*50)

# Linear Regression
print("\nTraining Linear Regression model...")
lr_income = LinearRegression()
lr_income.fit(X_train_income_scaled, y_train_income)
y_pred_lr_income = lr_income.predict(X_test_income_scaled)
lr_income_metrics = evaluate_model(y_test_income, y_pred_lr_income, "Linear Regression")
plot_actual_vs_predicted(y_test_income, y_pred_lr_income, 
                         "Linear Regression: Actual vs Predicted Income", 
                         os.path.join(output_dir, 'lr_income_prediction.png'))

# Random Forest Regressor
print("\nTraining Random Forest model...")
rf_income = RandomForestRegressor(n_estimators=100, random_state=42)
rf_income.fit(X_train_income_scaled, y_train_income)
y_pred_rf_income = rf_income.predict(X_test_income_scaled)
rf_income_metrics = evaluate_model(y_test_income, y_pred_rf_income, "Random Forest")
plot_actual_vs_predicted(y_test_income, y_pred_rf_income, 
                         "Random Forest: Actual vs Predicted Income", 
                         os.path.join(output_dir, 'rf_income_prediction.png'))
plot_feature_importance(rf_income, features, 
                       "Random Forest Feature Importance for Income Prediction", 
                       os.path.join(output_dir, 'rf_income_feature_importance.png'))

# Gradient Boosting Regressor
print("\nTraining Gradient Boosting model...")
gb_income = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_income.fit(X_train_income_scaled, y_train_income)
y_pred_gb_income = gb_income.predict(X_test_income_scaled)
gb_income_metrics = evaluate_model(y_test_income, y_pred_gb_income, "Gradient Boosting")
plot_actual_vs_predicted(y_test_income, y_pred_gb_income, 
                         "Gradient Boosting: Actual vs Predicted Income", 
                         os.path.join(output_dir, 'gb_income_prediction.png'))
plot_feature_importance(gb_income, features, 
                       "Gradient Boosting Feature Importance for Income Prediction", 
                       os.path.join(output_dir, 'gb_income_feature_importance.png'))

#===============================================
# MODEL 2: Predicting Disability Rate
#===============================================
print("\n" + "="*50)
print("MODEL 2: Predicting Disability Rate")
print("="*50)

# Linear Regression
print("\nTraining Linear Regression model...")
lr_disability = LinearRegression()
lr_disability.fit(X_train_disability_scaled, y_train_disability)
y_pred_lr_disability = lr_disability.predict(X_test_disability_scaled)
lr_disability_metrics = evaluate_model(y_test_disability, y_pred_lr_disability, "Linear Regression")
plot_actual_vs_predicted(y_test_disability, y_pred_lr_disability, 
                         "Linear Regression: Actual vs Predicted Disability Rate", 
                         os.path.join(output_dir, 'lr_disability_prediction.png'))

# Random Forest Regressor
print("\nTraining Random Forest model...")
rf_disability = RandomForestRegressor(n_estimators=100, random_state=42)
rf_disability.fit(X_train_disability_scaled, y_train_disability)
y_pred_rf_disability = rf_disability.predict(X_test_disability_scaled)
rf_disability_metrics = evaluate_model(y_test_disability, y_pred_rf_disability, "Random Forest")
plot_actual_vs_predicted(y_test_disability, y_pred_rf_disability, 
                         "Random Forest: Actual vs Predicted Disability Rate", 
                         os.path.join(output_dir, 'rf_disability_prediction.png'))
plot_feature_importance(rf_disability, features, 
                       "Random Forest Feature Importance for Disability Rate Prediction", 
                       os.path.join(output_dir, 'rf_disability_feature_importance.png'))

# Gradient Boosting Regressor
print("\nTraining Gradient Boosting model...")
gb_disability = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_disability.fit(X_train_disability_scaled, y_train_disability)
y_pred_gb_disability = gb_disability.predict(X_test_disability_scaled)
gb_disability_metrics = evaluate_model(y_test_disability, y_pred_gb_disability, "Gradient Boosting")
plot_actual_vs_predicted(y_test_disability, y_pred_gb_disability, 
                         "Gradient Boosting: Actual vs Predicted Disability Rate", 
                         os.path.join(output_dir, 'gb_disability_prediction.png'))
plot_feature_importance(gb_disability, features, 
                       "Gradient Boosting Feature Importance for Disability Rate Prediction", 
                       os.path.join(output_dir, 'gb_disability_feature_importance.png'))

#===============================================
# Model Hyperparameter Tuning for Best Model
#===============================================
print("\n" + "="*50)
print("Hyperparameter Tuning for the Best Model")
print("="*50)

# Based on the results, determine which model to tune
# Let's assume Gradient Boosting performed best for both tasks

print("\nTuning Gradient Boosting for Income Prediction...")
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# We'll use a smaller subset of the grid for demonstration
param_grid_small = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# Grid search for income prediction
grid_search_income = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid_small,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search_income.fit(X_train_income_scaled, y_train_income)
print("\nBest parameters for income prediction:")
print(grid_search_income.best_params_)

# Use the best model
best_gb_income = grid_search_income.best_estimator_
y_pred_best_gb_income = best_gb_income.predict(X_test_income_scaled)
best_gb_income_metrics = evaluate_model(y_test_income, y_pred_best_gb_income, "Tuned Gradient Boosting")
plot_actual_vs_predicted(y_test_income, y_pred_best_gb_income, 
                         "Tuned Gradient Boosting: Actual vs Predicted Income", 
                         os.path.join(output_dir, 'best_gb_income_prediction.png'))

print("\nTuning Gradient Boosting for Disability Rate Prediction...")
# Grid search for disability rate prediction
grid_search_disability = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid_small,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search_disability.fit(X_train_disability_scaled, y_train_disability)
print("\nBest parameters for disability rate prediction:")
print(grid_search_disability.best_params_)

# Use the best model
best_gb_disability = grid_search_disability.best_estimator_
y_pred_best_gb_disability = best_gb_disability.predict(X_test_disability_scaled)
best_gb_disability_metrics = evaluate_model(y_test_disability, y_pred_best_gb_disability, "Tuned Gradient Boosting")
plot_actual_vs_predicted(y_test_disability, y_pred_best_gb_disability, 
                         "Tuned Gradient Boosting: Actual vs Predicted Disability Rate", 
                         os.path.join(output_dir, 'best_gb_disability_prediction.png'))

#===============================================
# Summary of Results
#===============================================
print("\n" + "="*50)
print("Summary of Model Performance")
print("="*50)

print("\nIncome Prediction Models:")
models_income = {
    'Linear Regression': lr_income_metrics,
    'Random Forest': rf_income_metrics,
    'Gradient Boosting': gb_income_metrics,
    'Tuned Gradient Boosting': best_gb_income_metrics
}

for model, metrics in models_income.items():
    print(f"{model}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")

print("\nDisability Rate Prediction Models:")
models_disability = {
    'Linear Regression': lr_disability_metrics,
    'Random Forest': rf_disability_metrics,
    'Gradient Boosting': gb_disability_metrics,
    'Tuned Gradient Boosting': best_gb_disability_metrics
}

for model, metrics in models_disability.items():
    print(f"{model}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")

print("\nAnalysis complete! Models trained and evaluated successfully.")

# Save summary results to a file
summary_file = os.path.join(output_dir, 'model_performance_summary.txt')
with open(summary_file, 'w') as f:
    f.write("="*50 + "\n")
    f.write("Summary of Model Performance\n")
    f.write("="*50 + "\n\n")
    
    f.write("Income Prediction Models:\n")
    for model, metrics in models_income.items():
        f.write(f"{model}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}\n")
    
    f.write("\nDisability Rate Prediction Models:\n")
    for model, metrics in models_disability.items():
        f.write(f"{model}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}\n")

    f.write("\nBest Parameters for Income Prediction:\n")
    f.write(str(grid_search_income.best_params_) + "\n")
    
    f.write("\nBest Parameters for Disability Rate Prediction:\n")
    f.write(str(grid_search_disability.best_params_) + "\n")

print("Applying advanced feature engineering...")
# Add interaction terms and polynomial features
df['Age_Income_Interaction'] = df['Median age (years)'] * df['Income_Per_Capita']
df['Education_Income_Interaction'] = df['Education_Premium'] * df['Income_Per_Capita']
df['Age_Squared'] = df['Median age (years)']**2
df['Income_Per_Capita_Squared'] = df['Income_Per_Capita']**2
df['Dependency_Ratio'] = df['Percent_Under_18'] + df['Percent_65_plus']
df['Working_Age_Ratio'] = df['Percent_18_to_64'] / (100 - df['Percent_18_to_64'])

# Update features list
advanced_features = features + [
    'Age_Income_Interaction', 'Education_Income_Interaction',
    'Age_Squared', 'Income_Per_Capita_Squared',
    'Dependency_Ratio', 'Working_Age_Ratio'
]

# Handle outliers
def remove_outliers(X, y, n_sigmas=3):
    z_scores = stats.zscore(X)
    mask = np.all(np.abs(z_scores) < n_sigmas, axis=1)
    return X[mask], y[mask]

X_income_clean, y_income_clean = remove_outliers(X, y_income)
X_disability_clean, y_disability_clean = remove_outliers(X, y_disability)

# Apply robust scaling
robust_scaler = RobustScaler()
X_scaled_robust = robust_scaler.fit_transform(X_income_clean)

# Try the stacking ensemble
from sklearn.ensemble import StackingRegressor
stacking_regressor = StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ],
    final_estimator=LinearRegression()
)
stacking_regressor.fit(X_scaled_robust, y_income_clean)
y_pred_stack = stacking_regressor.predict(robust_scaler.transform(X_test_income))
stack_metrics = evaluate_model(y_test_income, y_pred_stack, "Stacking Ensemble")
