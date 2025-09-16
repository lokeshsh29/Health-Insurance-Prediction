
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
class TrainPipeline:
    def __init__(self):
        pass
    def run_training_pipeline():
        try:
            logging.info("Starting the training pipeline")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            # Step 3: Model Training
            model_trainer = ModelTrainer()
            result = model_trainer.initiate_model_trainer(train_arr, test_arr)
            print(result)

        except Exception as e:
            logging.info("Exception occurred in training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    TrainPipeline.run_training_pipeline()
