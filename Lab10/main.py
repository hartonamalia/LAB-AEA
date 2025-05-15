import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 1.
print("1.\n")

print("Importing data...")
df = pd.read_csv("data/chile.csv")

# Print information about data
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nData information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nEducation value counts:")
print(df['education'].value_counts())

print("\nVote value counts:")
print(df['vote'].value_counts())

print("\nSex value counts:")
print(df['sex'].value_counts())

print("\nRegion value counts:")
print(df['region'].value_counts())

print("\nMean values grouped by education (numeric columns only):")
print(df.groupby("education").mean(numeric_only=True))

print("\nMean age grouped by income:")
print(df.groupby("income")["age"].mean())

print("\nMean values grouped by education and vote (numeric columns only):")
print(df.groupby(['education', 'vote']).mean(numeric_only=True))

# 2.
print("2.\n")

variable = 'age'
print(f"\nAnalyzing variable: {variable}")

print("\nCentral tendency indicators:")
print(f"Mean: {df[variable].mean():.2f}")
print(f"Median: {df[variable].median():.2f}")
print(f"Mode: {df[variable].mode()[0]}")

print("\nDispersion indicators:")
print(f"Variance: {df[variable].var():.2f}")
print(f"Standard deviation: {df[variable].std():.2f}")
print(f"Range: {df[variable].max() - df[variable].min():.2f}")
print(f"Interquartile range: {df[variable].quantile(0.75) - df[variable].quantile(0.25):.2f}")
print(f"Min: {df[variable].min()}")
print(f"Max: {df[variable].max()}")

print("\nShape indicators:")
print(f"Skewness: {df[variable].skew():.2f}")
print(f"Kurtosis: {df[variable].kurtosis():.2f}")

plt.figure(figsize=(10, 6))
plt.hist(df[variable].dropna(), bins=10, edgecolor='black', alpha=0.7)
plt.title(f'Histogram of {variable}', fontsize=14)
plt.xlabel(variable, fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.savefig(f'{variable}_histogram.png')
print(f"\nHistogram saved as '{variable}_histogram.png'")

plt.figure(figsize=(8, 6))
plt.boxplot(df[variable].dropna(), vert=False)
plt.title(f'Boxplot of {variable}', fontsize=14)
plt.xlabel(variable, fontsize=12)
plt.grid(axis='x', alpha=0.75)
plt.savefig(f'{variable}_boxplot.png')
print(f"Boxplot saved as '{variable}_boxplot.png'")

# 3.
print("3.\n")

var1 = 'age'
var2 = 'statusquo'
print(f"\nAnalyzing correlation between {var1} and {var2}")

numerical_vars = ['age', 'income', 'statusquo']
correlation_matrix = df[numerical_vars].corr()
print("\nCorrelation matrix for numerical variables:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix for Numeric Variables', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")

corr, p_value = pearsonr(df[var1], df[var2])
print(f"\nPearson correlation between {var1} and {var2}:")
print(f"Correlation coefficient: {corr:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant correlation: {'Yes' if p_value < 0.05 else 'No'}")

plt.figure(figsize=(10, 6))
plt.scatter(df[var1], df[var2], alpha=0.6)
plt.title(f'{var1} vs {var2}', fontsize=14)
plt.xlabel(var1, fontsize=12)
plt.ylabel(var2, fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(f'{var1}_vs_{var2}.png')
print(f"\nScatter plot saved as '{var1}_vs_{var2}.png'")

print("\nAnalysis complete!")