import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngesionconfig:
    train_data_path=os.path.join("/config/workspace/artifacts",'train.csv')
    test_data_path=os.path.join("/config/workspace/artifacts",'test.csv')
    raw_data_path=os.path.join("/config/workspace/artifacts",'raw.csv')



class DataIngesion:
    def __init__(self):
        self.ingesion_config=DataIngesionconfig()

    def initiate_data_ingesion(self):
        logging.info("Data Ingesion method starts")

        try:
            df=pd.read_csv("/config/workspace/notebooks/data/gemstone.csv")
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingesion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingesion_config.raw_data_path,index=False)

            logging.info("train test split")

            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingesion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingesion_config.test_data_path,index=False,header=True)


            logging.info("Ingesion Of data is Completed")

            return(
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occured in Data Ingesion Config")
