"""
Missouri Data Analysis - Step 7: Advanced Nonlinear Models and Deep Learning
This script implements advanced modeling techniques including XGBoost, LightGBM, 
and neural networks for improved predictions of income and disability rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = os.path.join('analysis_output', 'advanced_ml_analysis')
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv("mo.csv")

# Reuse feature engineering from step6
def engineer_features(df):
    # First, convert relevant columns to numeric
    numeric_columns = [
        'Male', 'Female',
        'Under 5 years', '5 to 9 years', '10 to 14 years', '15 to 19 years',
        '20 to 24 years', '25 to 34 years', '35 to 44 years', '45 to 54 years',
        '55 to 59 years', '60 to 64 years', '65 to 74 years', '75 to 84 years',
        '85 years and over', 'Median age (years)',
        'Total housing units', 'Households Total',
        'Households Mean income (dollars)',
        "Total - MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings Bachelor's degree",
        'Tota - MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings!!High school graduate (includes equivalency)'
    ]
    
    # Print column types for debugging
    print("Converting columns to numeric...")
    
    # Convert to numeric, handling string and numeric data appropriately
    for col in numeric_columns:
        if col in df.columns:
            # Check if column is already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            # If column is object type (string), clean and convert
            elif pd.api.types.is_object_dtype(df[col]):
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            # For any other type, try direct conversion
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in dataframe")
    
    # Replace empty strings and blank values with NaN
    df = df.replace('', np.nan)
    
    print("Creating engineered features...")
    
    # Basic features (from step6)
    df['Percent_Male'] = df['Male'] / (df['Male'] + df['Female']) * 100
    df['Total_Population'] = df['Male'] + df['Female']
    df['Percent_Under_18'] = (df['Under 5 years'] + df['5 to 9 years'] + 
                             df['10 to 14 years'] + df['15 to 19 years'] * 0.8) / df['Total_Population'] * 100
    df['Percent_18_to_64'] = (df['15 to 19 years'] * 0.2 + df['20 to 24 years'] + 
                             df['25 to 34 years'] + df['35 to 44 years'] + 
                             df['45 to 54 years'] + df['55 to 59 years'] + 
                             df['60 to 64 years']) / df['Total_Population'] * 100
    df['Percent_65_plus'] = (df['65 to 74 years'] + df['75 to 84 years'] + 
                            df['85 years and over']) / df['Total_Population'] * 100
    
    # Housing and Income base features
    df['Housing_Density'] = df['Households Total'] / df['Total housing units']
    df['Income_Per_Capita'] = df['Households Mean income (dollars)'] / df['Households Total']
    
    # Handle education premium with error checking
    try:
        df['Education_Premium'] = (
            df["Total - MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings Bachelor's degree"] / 
            df['Tota - MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings!!High school graduate (includes equivalency)']
        )
    except Exception as e:
        print(f"Error calculating Education_Premium: {e}")
        # Use a fallback value
        df['Education_Premium'] = 1.5  # Default ratio
    
    # Now we can create interaction terms and transformations
    df['Age_Income_Interaction'] = df['Median age (years)'] * df['Income_Per_Capita']
    df['Education_Income_Interaction'] = df['Education_Premium'] * df['Income_Per_Capita']
    df['Age_Squared'] = df['Median age (years)']**2
    df['Income_Per_Capita_Squared'] = df['Income_Per_Capita']**2
    df['Dependency_Ratio'] = df['Percent_Under_18'] + df['Percent_65_plus']
    df['Working_Age_Ratio'] = df['Percent_18_to_64'] / (100 - df['Percent_18_to_64'])
    
    # Additional nonlinear transformations
    df['Log_Income'] = np.log1p(df['Households Mean income (dollars)'])
    df['Population_Density'] = df['Total_Population'] / df['Total housing units']
    df['Education_Age_Interaction'] = df['Education_Premium'] * df['Median age (years)']
    
    return df

# Feature selection
advanced_features = [
    'Median age (years)', 
    'Age_Squared',
    'Percent_Male',
    'Percent_Under_18', 
    'Percent_18_to_64', 
    'Percent_65_plus',
    'Housing_Density',
    'Education_Premium',
    'Income_Per_Capita',
    'Income_Per_Capita_Squared',
    'Age_Income_Interaction',
    'Education_Income_Interaction',
    'Dependency_Ratio',
    'Working_Age_Ratio',
    'Population_Density',
    'Education_Age_Interaction',
    'Log_Income'
]

# Prepare data
df = engineer_features(df)
X = df[advanced_features].copy()
y_income = df['Households Mean income (dollars)']
y_disability = df['Percent with a disability']

# Handle missing values and outliers
def prepare_data(X, y, outlier_threshold=3):
    # Remove rows with NaN
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X_clean = X[mask]
    y_clean = y[mask]
    
    # Remove outliers
    z_scores = stats.zscore(X_clean)
    outlier_mask = np.all(np.abs(z_scores) < outlier_threshold, axis=1)
    X_clean = X_clean[outlier_mask]
    y_clean = y_clean[outlier_mask]
    
    return X_clean, y_clean

# Prepare datasets
X_income, y_income = prepare_data(X, y_income)
X_disability, y_disability = prepare_data(X, y_disability)

# Split data
X_train_income, X_test_income, y_train_income, y_test_income = train_test_split(
    X_income, y_income, test_size=0.2, random_state=42
)

X_train_disability, X_test_disability, y_train_disability, y_test_disability = train_test_split(
    X_disability, y_disability, test_size=0.2, random_state=42
)

# Scale features
scaler = PowerTransformer(method='yeo-johnson')
X_train_income_scaled = scaler.fit_transform(X_train_income)
X_test_income_scaled = scaler.transform(X_test_income)

scaler_disability = PowerTransformer(method='yeo-johnson')
X_train_disability_scaled = scaler_disability.fit_transform(X_train_disability)
X_test_disability_scaled = scaler_disability.transform(X_test_disability)

# Model evaluation function
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

# Advanced Models
def train_advanced_models(X_train, X_test, y_train, y_test, target_name):
    results = {}
    
    # XGBoost
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results['XGBoost'] = evaluate_model(y_test, y_pred_xgb, "XGBoost")
    
    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb.fit(X_train, y_train)
    y_pred_lgb = lgb.predict(X_test)
    results['LightGBM'] = evaluate_model(y_test, y_pred_lgb, "LightGBM")
    
    # Neural Network
    nn = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=1000,
        random_state=42
    )
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    results['Neural Network'] = evaluate_model(y_test, y_pred_nn, "Neural Network")
    
    # Voting Ensemble
    voting_regressor = VotingRegressor([
        ('xgb', xgb),
        ('lgb', lgb),
        ('nn', nn)
    ])
    voting_regressor.fit(X_train, y_train)
    y_pred_voting = voting_regressor.predict(X_test)
    results['Voting Ensemble'] = evaluate_model(y_test, y_pred_voting, "Voting Ensemble")
    
    # Save results
    with open(os.path.join(output_dir, f'{target_name}_results.txt'), 'w') as f:
        f.write(f"Advanced Model Results for {target_name}\n")
        f.write("="*50 + "\n\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"R² = {metrics['r2']:.4f}\n")
            f.write(f"RMSE = {metrics['rmse']:.2f}\n")
            f.write(f"MAE = {metrics['mae']:.2f}\n\n")
    
    return results

# Train models for both targets
print("\nTraining advanced models for income prediction...")
income_results = train_advanced_models(
    X_train_income_scaled, X_test_income_scaled,
    y_train_income, y_test_income,
    "income"
)

print("\nTraining advanced models for disability prediction...")
disability_results = train_advanced_models(
    X_train_disability_scaled, X_test_disability_scaled,
    y_train_disability, y_test_disability,
    "disability"
)

print("\nAnalysis complete! Results saved in the advanced_ml_analysis directory.")
