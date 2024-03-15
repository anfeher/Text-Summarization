import os

from textSummarization.entity import DataValidationConfig
from textSummarization.logging import logger

class DataValidation:
    def __init__(self, config:DataValidationConfig) -> None:
        self.config = config

    def validate_all_files_exists(self) -> bool:
        try:
            validation_status = []

            all_files = os.listdir(os.path.join("artifacts","data_ingestion","samsum_dataset"))

            for file in self.config.ALL_REQUIRED_FILES:
                if file not in all_files:
                    validation_status.append(False)
                    logger.info(f"[{file}] doesn't exists")
                else:
                    validation_status.append(True)
                    logger.info(f"[{file}] exists")
                

            status = all(validation_status)
            logger.info(f"Validation status: {status}")
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {status}") 

            return status
        
        except Exception as e:
            raise e
