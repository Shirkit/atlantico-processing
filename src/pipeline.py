from sklearn.pipeline import Pipeline
from .transformers import ColumnSelector, Interpolator, ActivityFilter, WindowedAverageDownsampler

def create_preprocessing_pipeline(selected_columns=None, selected_columns_regex=None, sampling_strategy='interpolate'):
    """
    Creates a Scikit-learn pipeline for preprocessing.
    
    Args:
        selected_columns (list): List of columns to select.
        sampling_strategy (str): 'interpolate' or 'downsample'.
    """
    steps = [
        # 1. Filter out transient activities (ID=0)
        ('activity_filter', ActivityFilter(excluded_activities=[0])),
    ]
    
    # 2. Handle missing values / sampling
    if sampling_strategy == 'interpolate':
        steps.append(('interpolator', Interpolator(method='linear')))
    elif sampling_strategy == 'downsample':
        steps.append(('downsampler', WindowedAverageDownsampler(window_size=11, target_column='heart_rate')))
    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")
        
    # 3. Select specific columns if requested
    steps.append(('selector', ColumnSelector(columns=selected_columns, regex=selected_columns_regex)))
    
    return Pipeline(steps)
