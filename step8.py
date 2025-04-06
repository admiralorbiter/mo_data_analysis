"""
Missouri Data Analysis - Step 8: Model Application and Deployment
This script showcases our best-performing model (Voting Ensemble) for income prediction
as a proof of concept, including feature importance analysis and example predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import pickle
import os
import joblib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = os.path.join('analysis_output', 'model_application')
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess data
print("Loading and preprocessing data...")
df = pd.read_csv("mo.csv")

# Feature engineering function from step7
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
    
    # Replace empty strings and blank values with NaN
    df = df.replace('', np.nan)
    
    # Basic features
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
    
    # Advanced features
    df['Age_Income_Interaction'] = df['Median age (years)'] * df['Income_Per_Capita']
    df['Education_Income_Interaction'] = df['Education_Premium'] * df['Income_Per_Capita']
    df['Age_Squared'] = df['Median age (years)']**2
    df['Income_Per_Capita_Squared'] = df['Income_Per_Capita']**2
    df['Dependency_Ratio'] = df['Percent_Under_18'] + df['Percent_65_plus']
    df['Working_Age_Ratio'] = df['Percent_18_to_64'] / (100 - df['Percent_18_to_64'])
    df['Log_Income'] = np.log1p(df['Households Mean income (dollars)'])
    df['Population_Density'] = df['Total_Population'] / df['Total housing units']
    df['Education_Age_Interaction'] = df['Education_Premium'] * df['Median age (years)']
    
    return df

# Feature list (same as step7)
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

# Process the data
df_processed = engineer_features(df)
X = df_processed[advanced_features].copy()
y = df_processed['Households Mean income (dollars)']

# Handle missing values
mask = ~X.isna().any(axis=1) & ~y.isna()
X_clean = X[mask]
y_clean = y[mask]

# Remove outliers for better model training
z_scores = stats.zscore(X_clean)
outlier_mask = np.all(np.abs(z_scores) < 3, axis=1)
X_clean = X_clean[outlier_mask]
y_clean = y_clean[outlier_mask]

# Scale features
scaler = PowerTransformer(method='yeo-johnson')
X_scaled = scaler.fit_transform(X_clean)

# Create and train the best model (Voting Ensemble)
print("\nTraining the Voting Ensemble model...")
xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

nn = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=1000,
    random_state=42
)

# Initialize and train the Voting Ensemble
voting_ensemble = VotingRegressor([
    ('xgb', xgb),
    ('lgb', lgb),
    ('nn', nn)
])

voting_ensemble.fit(X_scaled, y_clean)

# Save the model and scaler for future use
print("Saving model and scaler...")
joblib.dump(voting_ensemble, os.path.join(output_dir, 'voting_ensemble_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'feature_scaler.pkl'))

# Feature importance analysis using individual models
print("\nAnalyzing feature importance...")
# Fit individual models for feature importance analysis
xgb.fit(X_scaled, y_clean)
lgb.fit(X_scaled, y_clean)

# Get feature importances from tree-based models
xgb_importances = xgb.feature_importances_
lgb_importances = lgb.feature_importances_

# Create a DataFrame to compare feature importances
feature_importance_df = pd.DataFrame({
    'Feature': advanced_features,
    'XGBoost Importance': xgb_importances,
    'LightGBM Importance': lgb_importances
})

# Sort by average importance
feature_importance_df['Average Importance'] = (
    feature_importance_df['XGBoost Importance'] + 
    feature_importance_df['LightGBM Importance']
) / 2
feature_importance_df = feature_importance_df.sort_values('Average Importance', ascending=False)

# Save feature importance table
feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

# Plot feature importance
plt.figure(figsize=(14, 8))
plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Average Importance'][:10])
plt.xlabel('Average Importance')
plt.title('Top 10 Most Important Features for Income Prediction')
plt.gca().invert_yaxis()  # Display with most important at the top
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_plot.png'))
plt.close()

# Example predictions on sample data
print("\nMaking predictions on sample data...")

# Get a sample of 5 ZCTAs with various income levels
sample_indices = []
income_ranges = [
    (0, 40000),       # Low income
    (40000, 60000),   # Lower-middle income
    (60000, 80000),   # Middle income
    (80000, 100000),  # Upper-middle income
    (100000, float('inf'))  # High income
]

for low, high in income_ranges:
    range_indices = y_clean[(y_clean >= low) & (y_clean < high)].index
    if len(range_indices) > 0:
        sample_indices.append(range_indices[0])

sample_X = X_clean.loc[sample_indices]
sample_y = y_clean.loc[sample_indices]

# Scale the sample data
sample_X_scaled = scaler.transform(sample_X)

# Make predictions
sample_pred = voting_ensemble.predict(sample_X_scaled)

# Create a comparison table
sample_comparison = pd.DataFrame({
    'Geography': df_processed.loc[sample_indices, 'Geography'] if 'Geography' in df_processed.columns else f"ZCTA {sample_indices}",
    'Actual Income': sample_y.values,
    'Predicted Income': sample_pred,
    'Difference': sample_y.values - sample_pred,
    'Percent Error': (np.abs(sample_y.values - sample_pred) / sample_y.values) * 100
})

# Save the sample comparison
sample_comparison.to_csv(os.path.join(output_dir, 'sample_predictions.csv'), index=False)

# Create a visualization of actual vs predicted for the samples
plt.figure(figsize=(10, 6))
x = np.arange(len(sample_comparison))
width = 0.35
plt.bar(x - width/2, sample_comparison['Actual Income'], width, label='Actual Income')
plt.bar(x + width/2, sample_comparison['Predicted Income'], width, label='Predicted Income')
plt.xlabel('Sample')
plt.ylabel('Household Income ($)')
plt.title('Actual vs Predicted Income for Sample ZCTAs')
plt.xticks(x, sample_comparison.index)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_prediction_comparison.png'))
plt.close()

# Scatter plot of all predictions
y_pred_all = voting_ensemble.predict(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(y_clean, y_pred_all, alpha=0.5)
plt.plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 'r--')
plt.xlabel('Actual Household Income')
plt.ylabel('Predicted Household Income')
plt.title('Actual vs Predicted Income (All Data)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_predictions_scatter.png'))
plt.close()

# Create an example function for making new predictions
def predict_income(data_dict):
    """
    Example function to predict income for a new ZCTA
    
    Parameters:
    data_dict (dict): Dictionary with feature values
    
    Returns:
    float: Predicted household income
    """
    # Create a dataframe with the input data
    input_df = pd.DataFrame([data_dict])
    
    # Apply feature engineering
    processed_df = engineer_features(input_df)
    
    # Extract features
    input_features = processed_df[advanced_features].values
    
    # Scale features
    input_scaled = scaler.transform(input_features)
    
    # Make prediction
    prediction = voting_ensemble.predict(input_scaled)[0]
    
    return prediction

# Example usage
print("\nDemonstrating prediction function with example data...")

example_input = {
    'Male': 5000,
    'Female': 5200,
    'Under 5 years': 650,
    '5 to 9 years': 700,
    '10 to 14 years': 750,
    '15 to 19 years': 800,
    '20 to 24 years': 850,
    '25 to 34 years': 1500,
    '35 to 44 years': 1400,
    '45 to 54 years': 1300,
    '55 to 59 years': 600,
    '60 to 64 years': 550,
    '65 to 74 years': 700,
    '75 to 84 years': 300,
    '85 years and over': 100,
    'Median age (years)': 38.5,
    'Total housing units': 4000,
    'Households Total': 3800,
    'Households Mean income (dollars)': None,  # We're predicting this
    "Total - MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings Bachelor's degree": 55000,
    'Tota - MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2023 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings!!High school graduate (includes equivalency)': 32000
}

try:
    predicted_income = predict_income(example_input)
    print(f"Predicted household income for example ZCTA: ${predicted_income:.2f}")
    
    # Save the example
    with open(os.path.join(output_dir, 'example_prediction.txt'), 'w') as f:
        f.write(f"Example Input Data:\n")
        for key, value in example_input.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nPredicted household income: ${predicted_income:.2f}")
        
except Exception as e:
    print(f"Error making example prediction: {e}")

print("\nAnalysis complete! Results saved in the model_application directory.")
