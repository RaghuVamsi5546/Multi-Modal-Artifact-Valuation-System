from src.utils.common import *
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_validation import DataValidation
from src.entity import DataTransformationConfig
import numpy as np
import pandas as pd 
import json 
import os

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_preprocessing(self):
        try:
            config = ConfigurationManager()
            data_preprocessed_config = config.get_data_preprocessed_config()
            data_validation_config = config.get_data_validation_config()
            data_transformation_config = config.get_data_transformation_config()
            schema_path = config.schema_path 

            # Load the NumPy arrays
            X_train_meta_np = np.load(os.path.join(data_transformation_config.root_dir, "X_train_meta.npy"), allow_pickle=True)
            X_val_meta_np   = np.load(os.path.join(data_transformation_config.root_dir, "X_val_meta.npy"), allow_pickle=True)
            X_test_meta_np  = np.load(os.path.join(data_transformation_config.root_dir, "X_test_meta.npy"), allow_pickle=True)

            y_train = np.load(os.path.join(data_transformation_config.root_dir, "y_train.npy"), allow_pickle=True)
            y_val = np.load(os.path.join(data_transformation_config.root_dir, "y_val.npy"), allow_pickle=True)
            y_test = np.load(os.path.join(data_transformation_config.root_dir, "y_test.npy"), allow_pickle=True)

            # FIX START: Load column names and convert NumPy arrays to DataFrames
            meta_columns_path = os.path.join(data_transformation_config.root_dir, 'meta_columns.json')
            with open(meta_columns_path, 'r') as f:
                meta_columns = json.load(f)

            X_train_meta_df = pd.DataFrame(X_train_meta_np, columns=meta_columns)
            X_val_meta_df = pd.DataFrame(X_val_meta_np, columns=meta_columns)
            X_test_meta_df = pd.DataFrame(X_test_meta_np, columns=meta_columns)
            # FIX END

            data_validation = DataValidation(
                config=data_validation_config,
                data_transformation_config=data_transformation_config,
                schema_path=schema_path
            )

            data_preprocessed = DataPreprocessing(
                config=data_preprocessed_config,
                data_validation=data_validation,
                data_transformation_config=data_transformation_config
            )
            
            # Pass the DataFrames to initiate_data_preprocess
            data_preprocessed.initiate_data_preprocess(
                X_train_meta=X_train_meta_df, 
                X_val_meta=X_val_meta_df, 
                X_test_meta=X_test_meta_df,
                y_train=y_train, 
                y_val=y_val, 
                y_test=y_test
            )

        except Exception as e:
            raise e