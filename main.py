from src.logging import logging
from src.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_2_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_3_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_4_data_preprocessing import DataPreprocessingTrainingPipeline

'''STAGE_NAME="Data Ingestion Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME="Data Validation Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    data_validation_pipeline=DataValidationTrainingPipeline()
    data_validation_pipeline.initiate_data_validation()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME="Data Transformation Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    data_transformation_pipeline=DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e'''

STAGE_NAME="Data Preprocessing Stage"

try:
    X_train_meta=None
    X_val_meta=None
    X_test_meta=None
    
    logging.info(f"{STAGE_NAME} initiated")
    data_preprocessing_pipeline=DataPreprocessingTrainingPipeline()
    data_preprocessing_pipeline.initiate_data_preprocessing()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e