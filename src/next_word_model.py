
import pathlib

import numpy as np

from keras.preprocessing.text import Tokenizer

import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (
    Embedding,
    Dense,
    LSTM
)
 

class NextWordModel(object):
    """
    expects X to be the sentence and y to be missing (correct) next word
    """

    def __init__(self, sequence_length, vocab_size, dimensions_to_represent_word=100, load_existing = True):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.dimensions_to_represent_word = dimensions_to_represent_word
        self.model_path = f"{pathlib.Path(__file__).parent.resolve()}/_objects/lstm.nextword.model"

        self.model = self.get_model()

        if load_existing == True:
            try:
                self._load_model()
                print("Loading existing model successful!")
            except Exception as e:
                print("Loading existing model failed:\n" + str(e))

    def train(self, X_train, y_train, epochs = 100, batch_size = 128, save = True):
        

        # Training may take a few hours without GPUs.
        self.model.fit(X_train, y_train, 
            batch_size = batch_size, 
            epochs = epochs
        )

        if save:
            self.model.save(self.model_path)


    def _load_model(self):
        self.model = tf.load(self.model_path)

    def get_model(self, verbose=True):
        model = Sequential()
        model.add(Embedding(
            self.vocab_size, 
            # self.sequence_length, 
            10,
            # input_length = self.sequence_length
            input_length = 1
            ))
        # unit =  memory cell
        # More memory cells and a deeper network may achieve better results.

        # computational expensiv
        model.add(LSTM(100, return_sequences = True))
        model.add(LSTM(100))
        model.add(Dense(100, activation= 'relu'))
        model.add(Dense(self.vocab_size, activation = 'softmax'))

        if verbose: 
            print(model.summary())

        model.compile(
            loss='categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy']
        )
        
        return model

    def predict(self, input):
        pass