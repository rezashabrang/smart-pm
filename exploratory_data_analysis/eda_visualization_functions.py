import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column):
    """Plot histogram for a specific column"""
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=10, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, column):
    """Plot boxplot for a specific column"""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

def plot_scatter(df, x_col, y_col):
    """Plot scatter plot for two columns"""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title(f'{x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical columns"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
