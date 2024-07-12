import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.preprocessing import FunctionTransformer
from src.utils import save_object


num_columns = ['age', 'bmi', 'children']
cat_columns = ['sex', 'smoker', 'region']

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('Getting data transsformation obj Initiated.')
            
            cat_pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OrdinalEncoder())
            ])
            num_pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            preprocessor = ColumnTransformer([
                ('num_pipe',num_pipeline,num_columns),
                ('cat_pipe',cat_pipeline,cat_columns)],
                remainder = 'passthrough'
            )

            return preprocessor

            logging.info('Data Pipeline has been completed.')

        except Exception as e:
            logging.info('Error occured in running get_data_transformation_obj')
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test splitted data from artifactss
            train_df = pd.read_csv('artifacts/train.csv')
            test_df =  pd.read_csv('artifacts/test.csv')
            
            logging.info('reading train and test data is completed.')
            logging.info(f'Train DataFrame Head: \n {train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n {test_df.head().to_string()}')


            logging.info('Concatenating Train and test csvs')
            

            preprocessing_obj = self.get_data_transformation_obj()

            target_columns_name=['expenses']
            drop_columns = ['expenses']
            # Concatinating Tarin and Test data 
            # df = pd.concat([train_df,test_df],axis=0)

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_columns_name]


            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_columns_name]

            ## Applying the preprocessing pipeline on Train and test data 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            
            logging.info('Obtaining preprocessing object')

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
           
            # logging.info(f'DataFrame Head: \n {df.head().to_string()}')
            

            # Defining Independent and dependent variables
            # features = df.drop(['expenses'],axis=1)
            # target = df['expenses']

            # #Applying these pipelines on dataset specifically
            # preprocessor_obj = self.get_data_transformation_obj()

            # features = preprocessor_obj.fit_transform(features)
            # # test_df1 = preprocessor_obj.fit_transform(test_df)
 
                       
            # all_columns = num_columns+cat_columns
            # # Independent features as Dataframe 
            # features = pd.DataFrame(features, columns = all_columns)

            
            # logging.info(f'Train DataFrame Head: \n {train_arr.head().to_string()}')
            # logging.info(f'Test DataFrame Head: \n {test_arr.head().to_string()}')

            
            
            logging.info("Applying preprocessing object on training and testing datasets.")

            
 
            #Saving the preprocessor.pkl object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )

            # logging.info("preprocessor pickle file saved.")
            logging.info('All sort of transformation has been done.')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path     
            )





        except Exception as e:
            logging.info('Error Occured in Initiating Data Transformation')
            raise CustomException(e,sys)