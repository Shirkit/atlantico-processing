import pandas as pd
import os
from typing import List, Optional, Generator

# Column definitions based on PAMAP2
IMU_COLS = ['temp', 
            'acc16_x', 'acc16_y', 'acc16_z', 
            'acc6_x', 'acc6_y', 'acc6_z', 
            'gyro_x', 'gyro_y', 'gyro_z', 
            'mag_x', 'mag_y', 'mag_z', 
            'ori_1', 'ori_2', 'ori_3', 'ori_4']

def get_column_names() -> List[str]:
    cols = ['timestamp', 'activityID', 'heart_rate']
    for pos in ['hand', 'chest', 'ankle']:
        cols.extend([f'{pos}_{c}' for c in IMU_COLS])
    return cols

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.columns = get_column_names()

    def load_subject(self, subject_id: int, chunksize: Optional[int] = None) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Loads data for a specific subject.
        """
        file_path = os.path.join(self.data_dir, f'subject{subject_id}.dat')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # The data is space-separated
        return pd.read_csv(
            file_path, 
            sep=r'\s+', 
            header=None, 
            names=self.columns,
            chunksize=chunksize
        )

    def load_all(self, subject_ids: List[int]) -> pd.DataFrame:
        """
        Loads and concatenates data for multiple subjects.
        Warning: This might consume a lot of memory.
        """
        dfs = []
        for sid in subject_ids:
            df = self.load_subject(sid)
            df['subject_id'] = sid # Add subject ID column
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
