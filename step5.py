import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory for correlation analysis
output_dir = Path('analysis_output/correlation_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
print("Loading data...")
df = pd.read_csv('mo.csv')

# Convert columns to numeric where possible
for col in df.columns:
    if col != 'Zipcode' and col != 'Geography' and col != 'Geographic Area Name':
        try:
            # Remove commas and convert to numeric
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        except:
            pass

# Select only numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation matrix
print("Calculating correlation matrix...")
correlation_matrix = df[numeric_cols].corr()

# Create a large figure for the heatmap
plt.figure(figsize=(20, 18))

# Create the heatmap
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f",
            vmin=-1, 
            vmax=1,
            square=True,
            cbar_kws={"shrink": .8})

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Add title
plt.title('Correlation Matrix of Missouri Census Data', pad=20, fontsize=16)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_file = output_dir / 'correlation_heatmap.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Correlation heatmap saved to {output_file}")

# Find strong correlations (absolute value > 0.7)
print("\nStrong correlations (|r| > 0.7):")
strong_correlations = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            strong_correlations.append((numeric_cols[i], numeric_cols[j], corr))

# Sort by absolute correlation value
strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

# Print top 10 strong correlations
for var1, var2, corr in strong_correlations[:10]:
    print(f"{var1} & {var2}: {corr:.3f}")

# Save correlation matrix to CSV for further analysis
correlation_matrix.to_csv(output_dir / 'correlation_matrix.csv')
print(f"\nCorrelation matrix saved to {output_dir / 'correlation_matrix.csv'}")
