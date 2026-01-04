
import os
import glob
import pandas as pd
import numpy as np
from src.loader import DataLoader
from src.pipeline import create_preprocessing_pipeline

PROTOCOL_DIR = './dataset/Protocol'
OPTIONAL_DIR = './dataset/Optional'
SUBJECT_ID = 105

def main():
    print(f"Loading data for subject {SUBJECT_ID}...")
    loader_protocol = DataLoader(PROTOCOL_DIR)
    loader_optional = DataLoader(OPTIONAL_DIR)
    
    df_protocol = loader_protocol.load_subject(SUBJECT_ID)
    df_optional = loader_optional.load_subject(SUBJECT_ID)
    
    if df_protocol is not None and df_optional is not None:
        df_merged = pd.concat([df_protocol, df_optional], ignore_index=True)
        print(f"Merged shape: {df_merged.shape}")
        
        # Create pipeline
        pipeline = create_preprocessing_pipeline(
            selected_columns=None,
            selected_columns_regex=r'^(activityID|heart_rate|.*_(temp|acc16_[xyz]|gyro_[xyz]|mag_[xyz]))$',
            sampling_strategy='downsample'
        )
        
        # We need to inspect the output of the pipeline step by step or just run it and check result
        # Let's run it
        print("Running pipeline...")
        processed = pipeline.fit_transform(df_merged)
        print(f"Processed shape: {processed.shape}")
        
        # Check for NaNs
        if processed.isnull().values.any():
            print("NaNs detected in processed data (before trimming)!")
            nan_rows = processed[processed.isnull().any(axis=1)]
            print(f"Number of rows with NaNs: {len(nan_rows)}")
            print("Indices of NaN rows:", nan_rows.index.tolist())
        
        # Simulate trimming
        MAX_OUTPUT_ROWS = 20000
        OUTPUT_SAMPLING_STRATEGY = 'uniform'
        
        if len(processed) > MAX_OUTPUT_ROWS:
            n = len(processed)
            keep = MAX_OUTPUT_ROWS
            if OUTPUT_SAMPLING_STRATEGY == 'uniform':
                indices = np.linspace(0, n - 1, num=keep, dtype=int)
                processed_trimmed = processed.iloc[indices].reset_index(drop=True)
                print(f"Trimmed to {keep} rows.")
                
                # Check row 15949 (index 15948)
                target_idx = 15948
                if target_idx < len(processed_trimmed):
                    row = processed_trimmed.iloc[target_idx]
                    print(f"\nRow {target_idx+1} (index {target_idx}) in trimmed data:")
                    print(row)
                    print("\nIs null?")
                    print(row.isnull())
                    
                    if row.isnull().any():
                        print("\nTHIS ROW HAS NANS!")
                
                # Check for any NaNs in trimmed data
                if processed_trimmed.isnull().values.any():
                    print("\nNaNs detected in TRIMMED data!")
                    nan_rows_trimmed = processed_trimmed[processed_trimmed.isnull().any(axis=1)]
                    print(f"Number of rows with NaNs in trimmed: {len(nan_rows_trimmed)}")
                    print("Indices in trimmed:", nan_rows_trimmed.index.tolist())

                # Search for the row from user logs
                # x: 0.244249, 0.812802, ...
                # heart_rate ~ 0.244249
                print("\nSearching for row with heart_rate ~ 0.244249...")
                matches = processed_trimmed[
                    (processed_trimmed['heart_rate'] > 0.244) & 
                    (processed_trimmed['heart_rate'] < 0.245)
                ]
                print(f"Found {len(matches)} matches.")
                if len(matches) > 0:
                    print(matches.head())
                    for idx, row in matches.iterrows():
                        if row.isnull().any():
                            print(f"Match at index {idx} HAS NANS!")
                            print(row)


            
    else:
        print("Could not load merged data.")

if __name__ == '__main__':
    main()
