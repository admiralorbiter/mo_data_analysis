import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style for better visualizations
plt.style.use('default')
sns.set_theme()

# Create output directory for plots
output_dir = Path('analysis_output/multivariate_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the processed dataset
df = pd.read_csv('analysis_output/7_processed_dataset.csv')

# Convert string numbers to float, handling any non-numeric values
def convert_to_numeric(df):
    for col in df.columns:
        try:
            # Replace any commas in numbers
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '')
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            continue
    return df

# Convert columns to numeric
df = convert_to_numeric(df)

# First, let's identify our actual columns and categorize them
columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Print available columns to help with debugging
print("Available numeric columns in dataset:")
print("\n".join(columns))

# Helper function to find columns containing certain keywords
def find_columns(keywords, columns):
    return [col for col in columns if any(keyword.lower() in col.lower() for keyword in keywords)]

# Categorize columns
demographic_cols = find_columns(['age', 'sex', 'male', 'female', 'population'], columns)
income_cols = find_columns(['income', 'earnings', 'poverty'], columns)
education_cols = find_columns(['education', 'school', 'bachelor', 'degree'], columns)
disability_cols = find_columns(['disability', 'difficulty'], columns)
work_cols = find_columns(['work', 'employed', 'labor'], columns)

# Define analysis pairs based on available columns
analysis_categories = {
    'Demographics': [],
    'Income and Work': [],
    'Education': [],
    'Disability': []
}

# Add demographic relationships
if demographic_cols and income_cols:
    analysis_categories['Demographics'].extend([
        (demographic_cols[0], income_cols[0])
    ])

# Add income relationships
if income_cols and work_cols:
    analysis_categories['Income and Work'].extend([
        (income_cols[0], work_cols[0])
    ])

# Add education relationships
if education_cols and income_cols:
    analysis_categories['Education'].extend([
        (education_cols[0], income_cols[0])
    ])

# Add disability relationships
if disability_cols and demographic_cols:
    analysis_categories['Disability'].extend([
        (disability_cols[0], demographic_cols[0])
    ])

def analyze_relationship(data, x_col, y_col):
    """Analyze relationship between two variables and return detailed statistics"""
    corr = data[x_col].corr(data[y_col])
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_col], data[y_col])
    
    # Determine relationship strength
    if abs(corr) < 0.3:
        strength = "weak"
    elif abs(corr) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if corr > 0 else "negative"
    significance = "significant" if p_value < 0.05 else "not significant"
    
    return {
        'correlation': corr,
        'strength': strength,
        'direction': direction,
        'p_value': p_value,
        'significance': significance,
        'slope': slope,
        'r_squared': r_value**2
    }

def create_enhanced_scatter_plot(data, x_col, y_col, output_dir, hue_col=None):
    """Create an enhanced scatter plot with regression line and statistics"""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    if hue_col:
        scatter = sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, alpha=0.6)
    else:
        scatter = sns.scatterplot(data=data, x=x_col, y=y_col, alpha=0.6)
    
    # Add regression line with confidence interval
    sns.regplot(data=data, x=x_col, y=y_col, scatter=False, 
                color='red', line_kws={'linestyle': '--'})
    
    # Calculate statistics
    stats_dict = analyze_relationship(data, x_col, y_col)
    
    # Add statistics text box
    stats_text = (
        f"Correlation: {stats_dict['correlation']:.2f}\n"
        f"R-squared: {stats_dict['r_squared']:.2f}\n"
        f"P-value: {stats_dict['p_value']:.3f}\n"
        f"Slope: {stats_dict['slope']:.2f}"
    )
    
    plt.figtext(0.95, 0.95, stats_text, 
                fontsize=10, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'Relationship between\n{x_col} and {y_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_name = f'scatter_{x_col}_vs_{y_col}'.replace(" ", "_")
    if hue_col:
        plot_name += f'_by_{hue_col}'
    plt.savefig(output_dir / f'{plot_name}.png')
    plt.close()
    
    return stats_dict

# Create detailed analysis summary
with open(output_dir / 'multivariate_analysis_summary.txt', 'w') as f:
    f.write("=== Multivariate Analysis Summary ===\n\n")
    
    # First, write column overview
    f.write("Available Data Categories:\n")
    f.write("=" * 50 + "\n")
    f.write(f"Demographic variables: {', '.join(demographic_cols)}\n")
    f.write(f"Income variables: {', '.join(income_cols)}\n")
    f.write(f"Education variables: {', '.join(education_cols)}\n")
    f.write(f"Disability variables: {', '.join(disability_cols)}\n")
    f.write(f"Work-related variables: {', '.join(work_cols)}\n\n")
    
    # Analyze each category
    for category, variable_pairs in analysis_categories.items():
        if variable_pairs:  # Only analyze categories with defined pairs
            f.write(f"\n{category} Analysis\n")
            f.write("=" * 50 + "\n")
            
            for x_col, y_col in variable_pairs:
                stats_dict = create_enhanced_scatter_plot(df, x_col, y_col, output_dir)
                
                f.write(f"\nRelationship: {x_col} vs {y_col}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Correlation coefficient: {stats_dict['correlation']:.3f}\n")
                f.write(f"Relationship strength: {stats_dict['strength']}\n")
                f.write(f"Direction: {stats_dict['direction']}\n")
                f.write(f"Statistical significance: {stats_dict['significance']}\n")
                f.write(f"R-squared value: {stats_dict['r_squared']:.3f}\n")
                f.write(f"P-value: {stats_dict['p_value']:.3e}\n")
                
                f.write("\nInterpretation:\n")
                f.write(f"- This indicates a {stats_dict['strength']} {stats_dict['direction']} relationship\n")
                f.write(f"- The relationship is statistically {stats_dict['significance']}\n")
                f.write(f"- {stats_dict['r_squared']*100:.1f}% of the variance in {y_col} can be explained by {x_col}\n")
                
                if stats_dict['significance'] == 'significant':
                    f.write(f"- For each unit increase in {x_col}, {y_col} changes by {stats_dict['slope']:.2f} units\n")
                
                f.write("\n")

    # Add correlation matrix analysis
    f.write("\nOverall Correlation Matrix Analysis\n")
    f.write("=" * 50 + "\n")
    
    # Select a subset of important columns for correlation analysis
    important_cols = demographic_cols[:3] + income_cols[:2] + education_cols[:2] + disability_cols[:2]
    correlation_matrix = df[important_cols].corr()
    
    # Find strongest correlations
    correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            correlations.append({
                'var1': correlation_matrix.index[i],
                'var2': correlation_matrix.columns[j],
                'correlation': correlation_matrix.iloc[i, j]
            })
    
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    f.write("\nTop 5 Most Important Correlations:\n")
    for i, corr in enumerate(correlations[:5], 1):
        f.write(f"{i}. {corr['var1']} and {corr['var2']}: {corr['correlation']:.3f}\n")
        f.write(f"   Interpretation: {'Strong' if abs(corr['correlation']) > 0.7 else 'Moderate' if abs(corr['correlation']) > 0.3 else 'Weak'} "
                f"{'positive' if corr['correlation'] > 0 else 'negative'} relationship\n")

print("Enhanced multivariate analysis complete! Check the summary file for detailed insights.")
