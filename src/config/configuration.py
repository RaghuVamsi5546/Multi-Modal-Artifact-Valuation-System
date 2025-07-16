from src.constants import CONFIG_FILE_PATH,SCHEMA_FILE_PATH
from src.utils.common import read_yaml,create_dictionaries,save_bin,load_bin

from src.entity import DataIngestionConfig,DataValidationConfig

class ConfigurationManager:
    def __init__(self,config_path=CONFIG_FILE_PATH,schema_path=SCHEMA_FILE_PATH):
        self.config=read_yaml(config_path)
        self.schema=read_yaml(schema_path)
        create_dictionaries([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_dictionaries([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            local_file=config.local_file
        )

        return data_ingestion_config
    
    def  get_data_validation_config(self)->DataValidationConfig:
        config=self.config.data_validation
        create_dictionaries([config.root_dir])
        self.schema_path = SCHEMA_FILE_PATH

        data_validation_config=DataValidationConfig(
            root_dir=config.root_dir,
            local_file=config.local_file,
            validation_status_path=config.validation_status_path
        )

        return data_validation_config