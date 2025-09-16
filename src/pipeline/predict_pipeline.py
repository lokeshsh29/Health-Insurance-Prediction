import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 sex: str,
                 age: int,
                 bmi: float,
                 children: int,
                 region: str,
                 smoker: str
                 ):
        self.sex = sex
        self.age = age
        self.bmi = bmi
        self.children = children
        self.region = region
        self.smoker = smoker

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "sex": [self.sex],
                "age": [self.age],
                "bmi": [self.bmi],
                "children": [self.children],
                "region": [self.region],
                "smoker": [self.smoker]
            }
            return pd.DataFrame(custom_data_input_dict, index=[0])
        except Exception as e:
            logging.info("Exception occurred in getting data frame")
            raise CustomException(e, sys)
    