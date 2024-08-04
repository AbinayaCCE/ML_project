import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and test input data')
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "LinearRegression":LinearRegression(),
                "DecisonTreeRegressor":DecisionTreeRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor()
            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            #to predict
            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
