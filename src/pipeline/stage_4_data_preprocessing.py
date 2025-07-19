from src.utils.common import *
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_validation import DataValidation
from src.entity import DataTransformationConfig
import numpy as np

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

            X_train_meta = np.load(os.path.join(data_transformation_config.root_dir, "X_train_meta.npy"), allow_pickle=True)
            X_val_meta   = np.load(os.path.join(data_transformation_config.root_dir, "X_val_meta.npy"), allow_pickle=True)
            X_test_meta  = np.load(os.path.join(data_transformation_config.root_dir, "X_test_meta.npy"), allow_pickle=True)

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
            
            data_preprocessed.initiate_data_preprocess(
                X_train_meta=X_train_meta, 
                X_val_meta=X_val_meta, 
                X_test_meta=X_test_meta
            )

        except Exception as e:
            raise e
