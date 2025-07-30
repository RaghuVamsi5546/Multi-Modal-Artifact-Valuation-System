import os
import mlflow
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        logging.info("Starting Data Transformation pipeline.")
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            mlops_config = config.get_mlops_config()

            os.environ["MLFLOW_TRACKING_URI"] = mlops_config.mlflow_uri
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlops_config.dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

            with mlflow.start_run():
                data_transformation = DataTransformation(config=data_transformation_config)
                # This method now returns the processed data, but we're not capturing it directly here
                # as it's saved to disk by the component itself for subsequent stages.
                data_transformation.initiate_data_transformation()

                mlflow.log_param("data_transformation_root_dir", str(data_transformation_config.root_dir))
                mlflow.log_param("text_features_column", data_transformation_config.text_features_column)
                mlflow.log_param("tfidf_max_features", data_transformation_config.tfidf_max_features)
                mlflow.log_param("count_vec_max_features", data_transformation_config.count_vec_max_features)
                mlflow.log_param("sentence_transformer_models", str(data_transformation_config.sentence_transformer_models))

                # Log artifacts created by DataTransformation
                mlflow.log_artifact(local_path=str(data_transformation_config.root_dir), artifact_path="data_transformation_artifacts")
                mlflow.log_artifact(local_path=str(data_transformation_config.text_preprocessor_artifacts_dir), artifact_path="text_preprocessor_artifacts")

            logging.info("Data Transformation pipeline finished.")
        except Exception as e:
            logging.error(f"Data Transformation pipeline failed: {e}")
            raise e
