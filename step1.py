import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os
from io import StringIO

# Create an output directory if it doesn't exist
output_dir = 'analysis_output'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv('mo.csv')

# Save initial overview to a text file
with open(f'{output_dir}/1_initial_overview.txt', 'w') as f:
    f.write("=== Dataset Overview ===\n")
    f.write("\nFirst few rows:\n")
    f.write(df.head().to_string())
    f.write(f"\n\nDataset shape: {df.shape}\n")
    
    f.write("\n=== Data Info ===\n")
    buffer = StringIO()
    df.info(buf=buffer)
    f.write(buffer.getvalue())

# Save descriptive statistics to CSV
df.describe().to_csv(f'{output_dir}/2_descriptive_statistics.csv')

# Check and save negative values information
with open(f'{output_dir}/3_negative_values.txt', 'w') as f:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    f.write("=== Checking for negative values ===\n")
    for col in numeric_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            f.write(f"Warning: {col} has {neg_count} negative values\n")

# Save skewness analysis
df.select_dtypes(include=[np.number]).skew().to_csv(f'{output_dir}/4_skewness_analysis.csv')

# Handle missing values
df = df.replace(['-', 'N'], pd.NA)

# Save missing values analysis
missing_info = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_info.to_csv(f'{output_dir}/5_missing_values_analysis.csv')

# Analyze and save imputation results
with open(f'{output_dir}/6_imputation_analysis.txt', 'w') as f:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        f.write(f"\nColumn: {col}\n")
        f.write(f"Missing: {missing_pct:.2f}%\n")
        
        if missing_pct > 0:
            if missing_pct < 5:
                skewness = df[col].skew()
                strategy = 'median' if abs(skewness) > 1 else 'mean'
                
                f.write(f"Strategy: Impute with {strategy} (skewness: {skewness:.2f})\n")
                imputer = SimpleImputer(strategy=strategy)
                df[f'{col}_imputed'] = imputer.fit_transform(df[[col]])
                
                f.write("\nBefore imputation:\n")
                f.write(df[col].describe().to_string())
                f.write("\n\nAfter imputation:\n")
                f.write(df[f'{col}_imputed'].describe().to_string())
                f.write("\n" + "="*50 + "\n")
            else:
                f.write("High missing rate - investigate before imputing\n")
                f.write("Distribution in non-missing data:\n")
                f.write(df[col].describe().to_string())
                f.write("\n" + "="*50 + "\n")

# Save final processed dataset
df.to_csv(f'{output_dir}/7_processed_dataset.csv', index=False)

print(f"""
Analysis files have been saved in the '{output_dir}' directory:
1. 1_initial_overview.txt - Basic dataset information
2. 2_descriptive_statistics.csv - Summary statistics
3. 3_negative_values.txt - Information about negative values
4. 4_skewness_analysis.csv - Skewness measures
5. 5_missing_values_analysis.csv - Missing data analysis
6. 6_imputation_analysis.txt - Detailed imputation results
7. 7_processed_dataset.csv - Final processed dataset
""")
