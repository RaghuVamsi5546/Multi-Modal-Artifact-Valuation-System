import pandas as pd
import os

from src.utils.common import *

from src.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    
    def read_data(self):
        try:
            df=pd.read_csv(self.config.local_file)
            os.makedirs(self.config.root_dir, exist_ok=True)
            df.to_csv(os.path.join(self.config.root_dir, 'data.csv'), index=False)
            return df
        except Exception as e:
            raise e