import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations - using a built-in style
plt.style.use('default')  # Changed from 'seaborn' to 'default'
sns.set_theme()  # This is the proper way to set seaborn defaults

# Create output directory for plots
output_dir = Path('analysis_output/univariate_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the processed dataset
df = pd.read_csv('analysis_output/7_processed_dataset.csv')

def create_distribution_plots(data, column, output_dir):
    """Create and save distribution plots for a numerical column"""
    # Create a figure with 2 subplots (histogram with KDE and boxplot)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram with KDE
    sns.histplot(data=data, x=column, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {column}')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Count')
    
    # Boxplot
    sns.boxplot(y=data[column], ax=ax2)
    ax2.set_title(f'Boxplot of {column}')
    
    # Add descriptive statistics as text
    stats = data[column].describe()
    stats_text = (f'Mean: {stats["mean"]:.2f}\n'
                 f'Median: {stats["50%"]:.2f}\n'
                 f'Std: {stats["std"]:.2f}\n'
                 f'Skewness: {data[column].skew():.2f}')
    
    plt.figtext(0.95, 0.95, stats_text, fontsize=10, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{column}_distribution.png')
    plt.close()

# List of numerical columns to analyze
# We'll identify them based on data types and column names
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Categorize columns
age_columns = [col for col in numeric_columns if 'age' in col.lower()]
income_columns = [col for col in numeric_columns if 'income' in col.lower()]
earnings_columns = [col for col in numeric_columns if 'earning' in col.lower()]
disability_columns = [col for col in numeric_columns if 'disability' in col.lower()]
education_columns = [col for col in numeric_columns if 'education' in col.lower()]

# Create a summary file
with open(output_dir / 'analysis_summary.txt', 'w') as f:
    f.write("=== Univariate Analysis Summary ===\n\n")
    
    # Analyze each category of variables
    for category, columns in [
        ("Age-related", age_columns),
        ("Income-related", income_columns),
        ("Earnings-related", earnings_columns),
        ("Disability-related", disability_columns),
        ("Education-related", education_columns)
    ]:
        f.write(f"\n{category} Variables:\n")
        f.write("=" * 50 + "\n")
        
        for col in columns:
            # Create distribution plots
            create_distribution_plots(df, col, output_dir)
            
            # Write summary statistics
            stats = df[col].describe()
            skewness = df[col].skew()
            
            f.write(f"\nVariable: {col}\n")
            f.write(f"Mean: {stats['mean']:.2f}\n")
            f.write(f"Median: {stats['50%']:.2f}\n")
            f.write(f"Std Dev: {stats['std']:.2f}\n")
            f.write(f"Skewness: {skewness:.2f}\n")
            f.write(f"IQR: {stats['75%'] - stats['25%']:.2f}\n")
            f.write(f"Range: {stats['max'] - stats['min']:.2f}\n")
            f.write("-" * 30 + "\n")

# Create correlation heatmap for each category
for category, columns in [
    ("Age-related", age_columns),
    ("Income-related", income_columns),
    ("Earnings-related", earnings_columns),
    ("Disability-related", disability_columns),
    ("Education-related", education_columns)
]:
    if len(columns) > 1:  # Only create heatmap if there are multiple columns
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[columns].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation Heatmap: {category} Variables')
        plt.tight_layout()
        plt.savefig(output_dir / f'{category.lower()}_correlation_heatmap.png')
        plt.close()

print(f"""
Univariate analysis complete! Results saved in '{output_dir}':
1. Distribution plots (histogram + KDE and boxplot) for each numerical variable
2. Correlation heatmaps for each category of variables
3. Detailed statistical summary in 'analysis_summary.txt'

The analysis includes:
- Central tendency measures (mean, median)
- Spread measures (standard deviation, IQR)
- Skewness analysis
- Visual distribution analysis
- Correlation analysis within variable categories
""")
