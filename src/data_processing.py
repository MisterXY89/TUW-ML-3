
import numpy as np
import pandas as pd

import string
import pickle
import pathlib

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical


class Process(object):

    def __init__(self, data_loader, test_size = 0.2, sample_factor = 0.8):
        self.tokenizer = Tokenizer()
        self.data_loader = data_loader
        self.file_path = pathlib.Path(__file__).parent.resolve()

        self.test_size = test_size
        self.sample_factor = sample_factor

    def _filter_links(self, df):
        url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        df["hasLink"] = df.text.str.contains(url_regex)
        df = df.query("hasLink == False").drop(columns=["hasLink"])
        df = df.reset_index(drop=True)
        return df

    def _tokenize(self, df):
        text_data = " ".join(list(df.text))
        self.tokenizer.fit_on_texts([text_data])
        with open(f"{self.file_path}/_objects/tokenizer.pkl", "wb") as fi:
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

        tokens = self._tokenize(data)
        sequences = self._cut_sequences(tokens)

        X, y = self._get_Xy(sequences)        

        return X, y

    def _test_train_split(self, data):
        data = data.sample(frac = self.sample_factor)

        test = data.sample(frac = self.test_size)
        train = data[~data.index.isin(test.index)]

        return train, test 

    def process(self, store=True, force=False, sequence_length=2):
        self.sequence_length = sequence_length
        data, is_raw = self.data_loader.load_data(force=force)        
        
        if not force:
            if is_raw:                 
                print("No preprocessed file to load - <force> will be ignored.")
                data = data.sample(5000)
            else:
                print("Data loading complete")
                return data
        
        train, test = self._test_train_split(data)
        X_test, y_test = self._process(test)
        X_train, y_train = self._process(train)

        # if store: 
        #     self.data_loader.store_processed_data(data)

        return X_train, X_test, y_train, y_test
        
        # {
        #     "train": {
        #         "X": X_train,
        #         "y": y_train
        #     },
        #     "test": {
        #         "X": X_test,
        #         "y": y_test
        #     }
        # }
