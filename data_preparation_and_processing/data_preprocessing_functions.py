import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

def handle_missing_data(df):
    """Impute missing values with mean for numerical columns"""
    df.fillna(df.mean(), inplace=True)
    return df

def remove_duplicates(df):
    """Remove duplicate rows"""
    df.drop_duplicates(inplace=True)
    return df

def detect_outliers(df):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(df))
    return df[(z_scores < 3).all(axis=1)]  # Keep only rows with Z-score < 3

def scale_data(df):
    """Scale numerical features using StandardScaler"""
    scaler = StandardScaler()
    df[['Budgeted Cost', 'Actual Cost']] = scaler.fit_transform(df[['Budgeted Cost', 'Actual Cost']])
    return df
