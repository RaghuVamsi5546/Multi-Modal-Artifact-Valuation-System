from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from src.utils.common import read_yaml, create_dictionaries, save_bin, load_bin

from src.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, DataPreprocessingConfig,ModelTrainerConfig,ModelEvaluationConfig

class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH, schema_path=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_path)
        self.schema = read_yaml(schema_path)
        create_dictionaries([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_dictionaries([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_file=config.local_file
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_dictionaries([config.root_dir])
        self.schema_path = SCHEMA_FILE_PATH

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            local_file=config.local_file,
            validation_status_path=config.validation_status_path
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
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

    def get_data_preprocessed_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        create_dictionaries([config.root_dir])
        self.schema_path = SCHEMA_FILE_PATH

        data_preprocessed_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            imputation_strategy_numeric=config.imputation_strategy_numeric,
            imputation_strategy_categorical=config.imputation_strategy_categorical,
            scaler_type=config.scaler_type,
            encoder_type=config.encoder_type,
            transformed_data_dir=config.transformed_data_dir
        )

        return data_preprocessed_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_dictionaries([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_dir=config.train_data_dir,
            validation_data_dir=config.validation_data_dir,
            test_data_dir=config.test_data_dir,
            preprocessor_path=config.preprocessor_path,
            text_vectorizer_path=config.text_vectorizer_path,
            target_column=config.target_column,
            model_params=config.model_params,
            metric_file_path=config.metric_file_path,
            trained_model_dir=config.trained_model_dir
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_dictionaries([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_dir=config.test_data_dir,
            trained_model_dir=config.trained_model_dir,
            metric_file_path=config.metric_file_path
        )

        return model_evaluation_config