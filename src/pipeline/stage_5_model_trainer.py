from src.utils.common import *
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer # Import the ModelTrainer class
from src.entity import ModelTrainerConfig # Import ModelTrainerConfig


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        try:
            logging.info("Starting model training pipeline...")
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()

            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train_model() # This method orchestrates Goal 1 and Goal 2 training

            logging.info("Model training pipeline completed successfully.")

        except Exception as e:
            logging.error(f"Model training pipeline failed: {e}")
            raise e
