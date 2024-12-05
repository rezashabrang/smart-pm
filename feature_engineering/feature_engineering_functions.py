import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_project_duration(df):
    """Create a new feature for project duration"""
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    df['Project Duration'] = (df['End Date'] - df['Start Date']).dt.days
    return df

def create_cost_overrun(df):
    """Create a new feature for cost overrun"""
    df['Cost Overrun'] = df['Actual Cost'] / df['Budgeted Cost']
    return df

def scale_features(df, features):
    """Scale numerical features"""
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def one_hot_encode(df, column):
    """One-hot encode categorical features"""
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[[column]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.categories_[0])
    df = pd.concat([df, encoded_df], axis=1)
    return df
