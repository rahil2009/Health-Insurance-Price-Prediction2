# Baic Import
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import  ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils  import save_object
from  src.utils import evaluate_model
from dataclasses import dataclass

# Defining a Dataclass
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting dataset into the dependent and independent feautes')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
                

            models = {
                'Linear_Reg' : LinearRegression(),
                'DTree' : DecisionTreeRegressor(min_samples_leaf=.0001),
                'RForest' : RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001),
                'RForest1' : RandomForestRegressor(n_estimators=100,random_state=329,min_samples_leaf=.0001),
                'ETree' : ExtraTreesRegressor(n_estimators = 100),
                'GradientBoosting' :GradientBoostingRegressor(n_estimators=70,learning_rate=0.1)

            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('='*40)
            print(f"model report:{model_report}")
            logging.info(f"MOdel_Report : {model_report}")
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )



        except Exception as e:
            logging.info('Error occured in Initiate_model_training part')
            raise CustomException(e,sys)