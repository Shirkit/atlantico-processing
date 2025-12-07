from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from the DataFrame.
    """
    def __init__(self, columns=None, regex=None):
        self.columns = columns
        self.regex = regex

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns:
            return X[self.columns]
        if self.regex:
            return X.filter(regex=self.regex)
        return X

class Interpolator(BaseEstimator, TransformerMixin):
    """
    Interpolates missing values.
    """
    def __init__(self, method='linear', limit_direction='both'):
        self.method = method
        self.limit_direction = limit_direction

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure we are working with a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.interpolate(method=self.method, limit_direction=self.limit_direction)

class WindowedAverageDownsampler(BaseEstimator, TransformerMixin):
    """
    Downsamples data by taking a windowed average around rows with valid values in a target column.
    """
    def __init__(self, window_size=11, target_column='heart_rate'):
        self.window_size = window_size
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if self.target_column not in X.columns:
            return X
        
        # Identify rows with valid target data
        valid_mask = X[self.target_column].notna()
        
        # Columns to exclude from averaging (categorical or IDs)
        no_avg_cols = ['activityID', 'subject_id', 'timestamp']
        cols_to_avg = [c for c in X.columns if c not in no_avg_cols]
        
        # Calculate rolling mean
        # min_periods=1 ensures we get a value even if some neighbors are NaN
        rolled = X[cols_to_avg].rolling(
            window=self.window_size, 
            center=True, 
            min_periods=1
        ).mean()
        
        # Restore non-averaged columns from the original rows
        for c in no_avg_cols:
            if c in X.columns:
                rolled[c] = X[c]
        
        # Filter to keep only the rows where target_column was originally valid
        result = rolled[valid_mask].copy()
        
        # Reorder columns to match input
        result = result[X.columns]
        
        return result

class ActivityFilter(BaseEstimator, TransformerMixin):
    """
    Filters rows based on activity ID.
    Activity ID 0 is usually 'transient' or 'null' class in PAMAP2.
    """
    def __init__(self, excluded_activities=[0]):
        self.excluded_activities = excluded_activities

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if 'activityID' in X.columns:
            return X[~X['activityID'].isin(self.excluded_activities)]
        return X
