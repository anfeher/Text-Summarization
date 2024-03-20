from textSummarization.config.configuration import ConfigurationManager
from textSummarization.components.model_trainer import ModelTrainer
from textSummarization.logging import logger

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()