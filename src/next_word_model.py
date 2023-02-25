
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

# from .data_processing import Process
 

class NextWordModel(object):
    """
    expects X to be the sentence and y to be missing (correct) next word    
    """

    def __init__(self, dimensions_to_represent_word=100, load_existing = False, processor = None, model_name = "", optimizer = "adam", loss = "categorical_crossentropy"):
        self.processor = processor
        self.sequence_length = self.processor.sequence_length,
        self.vocab_size = self.processor.vocab_size,        

        self.dimensions_to_represent_word = dimensions_to_represent_word
        self.model_name = "lstm.nextword.model" if not model_name else model_name
        self.model_path = f"{pathlib.Path(__file__).parent.resolve()}/_objects/{self.model_name}"

        if load_existing == True:
            try:
                self._load_model()
                print(f"Loading existing model (model name = {self.model_name}) successful!")
            except Exception as e:
                print(f"Loading existing model (model name = {self.model_name}) failed:\n" + str(e))                
                self.model = self.get_model(optimizer = optimizer, loss = loss)
        else:
            self.model = self.get_model(optimizer = optimizer, loss = loss)

    def train(self, X_train, y_train, epochs = 100, batch_size = 128, save = True):
        
        # Training may take a few hours without GPUs.
        self.model.fit(X_train, y_train, 
            batch_size = batch_size, 
            epochs = epochs
        )

        if save:
            self.model.save(self.model_path)


    def _load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def get_model(self, verbose=True, loss = 'categorical_crossentropy', optimizer = "adam"):
        model = Sequential()
        print(self.vocab_size)
        model.add(
            Embedding(
                self.vocab_size[0], 
                # self.sequence_length, 
                10,
                # input_length = self.sequence_length
                input_length = 1
            )
        )

        # unit =  memory cell
        # More memory cells and a deeper network may achieve better results.

        # computational expensiv
        model.add(LSTM(100, return_sequences = True))
        model.add(LSTM(100))
        model.add(Dense(100, activation= 'relu'))
        model.add(Dense(self.vocab_size[0], activation = 'softmax'))

        if verbose: 
            print(model.summary())

        model.compile(
            loss=loss, 
            optimizer=optimizer, 
            metrics=['accuracy']
        )
        
        return model

    def predict(self, inp):
        if isinstance(inp, str):
            text = inp.split(" ")
            text = text[-1]
            text = ''.join(text)
            
            sequence = self.processor.tokenizer.texts_to_sequences(text)[0]
            sequence = np.array(sequence)
        
        preds = self.model.predict(sequence, verbose = 0)
        preds = np.argmax(preds, axis=1)
        predicted_word = ""

        predicted_word = [key for key, value in self.processor.tokenizer.word_index.items() if value == preds][0]
    
        # for key, value in self.processor.tokenizer.word_index.items():
        #     # print(preds)
        #     if value == preds:
        #         predicted_word = key
        #         break
        
        # print(predicted_word)
        return predicted_word