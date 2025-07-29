from src.utils.common import *
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()

            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.initiate_model_evaluation()
        except Exception as e:
            logging.error(f"Model evaluation pipeline failed: {e}")
            raise e
