import pandas as pd

from src.utils.common import *

from src.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    
    def read_data(self):
        try:
            df=pd.read_csv(self.config.local_file)
        except Exception as e:
            raise e