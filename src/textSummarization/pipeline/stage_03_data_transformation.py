from textSummarization.config.configuration import ConfigurationManager
from textSummarization.components.data_transformation import DataTransformation
from textSummarization.logging import logger

class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()