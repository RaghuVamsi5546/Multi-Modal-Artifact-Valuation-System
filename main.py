from src.logging import logging
from src.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_2_data_validation import DataValidationTrainingPipeline

STAGE_NAME="Data Ingestion Stage"

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