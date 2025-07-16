from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.constants import SCHEMA_FILE_PATH
from src.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config=ConfigurationManager()
        data_validation_config=config.get_data_validation_config()
        data_validation=DataValidation(config=data_validation_config,schema=SCHEMA_FILE_PATH)
        data_validation.validate_columns()