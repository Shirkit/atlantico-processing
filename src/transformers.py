from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    def __init__(self, method='linear', limit_direction='both', exclude_columns=None):
        self.method = method
        self.limit_direction = limit_direction
        self.exclude_columns = exclude_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure we are working with a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if not self.exclude_columns:
            return X.interpolate(method=self.method, limit_direction=self.limit_direction)
            
        # Interpolate only non-excluded columns
        cols_to_interp = [c for c in X.columns if c not in self.exclude_columns]
        if not cols_to_interp:
            return X
            
        X_interp = X.copy()
        X_interp[cols_to_interp] = X[cols_to_interp].interpolate(method=self.method, limit_direction=self.limit_direction)
        return X_interp

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

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scales feature columns using StandardScaler, excluding specified columns (e.g. labels, timestamps).
    """
    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns or []
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Identify columns to scale
        if isinstance(X, pd.DataFrame):
            cols_to_scale = [c for c in X.columns if c not in self.exclude_columns]
            if cols_to_scale:
                self.scaler.fit(X[cols_to_scale])
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            cols_to_scale = [c for c in X.columns if c not in self.exclude_columns]
            if cols_to_scale:
                X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])
        return X

class FeatureScalerMinMax(BaseEstimator, TransformerMixin):
    """
    Scales feature columns using StandardScaler, excluding specified columns (e.g. labels, timestamps).
    """
    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns or []
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        # Identify columns to scale
        if isinstance(X, pd.DataFrame):
            cols_to_scale = [c for c in X.columns if c not in self.exclude_columns]
            if cols_to_scale:
                self.scaler.fit(X[cols_to_scale])
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            cols_to_scale = [c for c in X.columns if c not in self.exclude_columns]
            if cols_to_scale:
                X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])
        return X
