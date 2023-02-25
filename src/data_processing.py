
import numpy as np
import pandas as pd

import string
import pickle
import pathlib

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


class Process(object):

    def __init__(self, data_loader, test_size = 0.2, sample_factor = 0.8, model_name = ""):        
        self.tok_name = "tokenizer" if not model_name else f"tokenizer.{model_name}"
        print(self.tok_name)
        self.file_path = pathlib.Path(__file__).parent.resolve()

        try:
            with open(f"{self.file_path}/_objects/{self.tok_name}.pkl", "rb") as fi:
                self.tokenizer = pickle.load(fi)
            self.load_tokenizer = True
        except Exception as e:            
            self.tokenizer = Tokenizer()
            self.load_tokenizer = False

        self.data_loader = data_loader        

        self.test_size = test_size
        self.sample_factor = sample_factor

    def _filter_links(self, df):
        url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        df["hasLink"] = df.text.str.contains(url_regex)
        df = df.query("hasLink == False").drop(columns=["hasLink"])
        df = df.reset_index(drop=True)
        return df

    def tokenize(self, df):      

        if isinstance(df, pd.DataFrame):
            text_data = " ".join(list(df.text))
        else:
            text_data = df


        self.tokenizer.fit_on_texts([text_data])

        if not self.load_tokenizer:
            with open(f"{self.file_path}/_objects/{self.tok_name}.pkl", "wb") as fi:
                pickle.dump(self.tokenizer, fi)
                
        token_sequences = self.tokenizer.texts_to_sequences([text_data])[0]
        self.vocab_size = len(self.tokenizer.word_index) + 1
        return token_sequences
        
    def _cut_sequences(self, token_list):
        # self.sequence_length currently not used!!
        sequences = []

        for i in range(1, len(token_list)):
            words = token_list[i-1:i+1]
            # line = ' '.join(words)
            sequences.append(words)
            
        sequences = np.array(sequences)
        return sequences

    def _get_Xy(self, sequences):
        X = []
        y = []

        for i in sequences:
            X.append(i[0])
            y.append(i[1])
            
        X = np.array(X)
        y = np.array(y)

        y = to_categorical(y, num_classes=self.vocab_size)
        return X, y

    def _process(self, data):
        # remove all tweets with links
        data = self._filter_links(data)
        # remove all punctuation
        data["text"] = data.text.apply(lambda x: " " if x in string.punctuation else x)
        # remove all numbers and special signs like %, § ...
        data["text"] = data.text.apply(lambda x: " ".join([w for w in x.split() if w.isalpha()]))
        # lower all and remove trainling/leading whitespaces
        data["text"] = data.text.apply(lambda x: x.lower().strip())        

        tokens = self.tokenize(data)
        sequences = self._cut_sequences(tokens)

        X, y = self._get_Xy(sequences)        

        return X, y
    

    def process(self, store=True, force=False, sequence_length=2, random_state=1180):
        self.random_state = random_state
        self.sequence_length = sequence_length
        data, is_raw = self.data_loader.load_data(force=force)   

        data = data.sample(frac = self.sample_factor, random_state = self.random_state)     
        
        if not force:
            if is_raw:                 
                print("No preprocessed file to load - <force> will be ignored.")
                data = data.sample(5000)
            else:
                print("Data loading complete")
                return data
                
        # TODO: tokenize for all, then process for each!!!!
        X, y = self._process(data)

        # X_train, X_test, y_train, y_test = self._test_train_split(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size = self.test_size, 
            random_state = 1180
        )

        return X_train, X_test, y_train, y_test