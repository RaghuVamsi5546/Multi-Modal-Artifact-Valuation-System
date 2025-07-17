from src.constants import CONFIG_FILE_PATH,SCHEMA_FILE_PATH
from src.utils.common import read_yaml,create_dictionaries,save_bin,load_bin

from src.entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig

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
    
    def get_data_validation_config(self)->DataValidationConfig:
        config=self.config.data_validation
        create_dictionaries([config.root_dir])
        self.schema_path = SCHEMA_FILE_PATH

        data_validation_config=DataValidationConfig(
            root_dir=config.root_dir,
            local_file=config.local_file,
            validation_status_path=config.validation_status_path
        )

        return data_validation_config

    def get_data_transformation_config(self)-> DataTransformationConfig:
        config = self.config.data_transformation
        create_dictionaries([config.root_dir])
        tfidf_ngram_range_value = tuple(config.tfidf_n_gram_range)
        count_vec_ngram_range_value = tuple(config.count_vec_ngram_range)

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            validation_data_path=config.validation_data_path,
            test_data_path=config.test_data_path,
            text_features_column=config.text_features_column,
            tfidf_max_features=config.tfidf_max_features,
            tfidf_n_gram_range=tfidf_ngram_range_value,
            count_vec_max_features=config.count_vec_max_features,
            count_vec_ngram_range=count_vec_ngram_range_value,
            sentence_transformer_models=config.sentence_transformer_models,
            text_preprocessor_artifacts_dir=config.text_preprocessor_artifacts_dir
        )

        return data_transformation_config