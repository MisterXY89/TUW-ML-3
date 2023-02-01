
import pickle
import pathlib

import numpy as np
import pandas as pd

class DataLoader():

    def __init__(self, external=False) -> None:
        self.file_path = pathlib.Path(__file__).parent.resolve()
        if external: 
            self.file_path = "https://github.com/MisterXY89/TUW-ML-3/blob/main/src"
        self.DATA_DIR = f"{self.file_path}/data"

        self.max_data_load_tries = 3

    def load_data(self, force=False, load_try=0):
        if load_try >= self.max_data_load_tries:
            raise Exception(f"Failed loading data - tried {self.max_data_load_tries} times")
            
        if force:
            df = pd.read_csv(f"{self.DATA_DIR}/trump_tweets.csv")
            return df.query("isRetweet == 'f'")[["id", "text"]], True
        
        processed_data = self._load_processed_data()        
        if not isinstance(processed_data, dict) and not force:
            return self.load_data(force=True, load_try=load_try+1)

        return processed_data, False
    
    def _load_processed_data(self):
        try:
            with open(f"{self.file_path}/_objects/data.pkl", "rb") as fi:
                return pickle.load(fi)
        except Exception as e:
            print(e)
            return False

    
    def store_processed_data(self, data):
        try:
            with open(f"{self.file_path}/_objects/data.pkl", "wb") as fi:
                pickle.dump(data, fi)
        except Exception as e:
            print(e)
            return False

