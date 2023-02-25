
import argparse

from src.data_loading import DataLoader
from src.data_processing import Process

from src.next_word_model import NextWordModel

def train_model(sample_factor, model_name, epochs, batch_size):
    dl = DataLoader()
    process = Process(
        dl,
        sample_factor = 0.95
    )

    # obtain the test and train data
    X_train, X_test, y_train, y_test = process.process(force = True)

    # initalize the model
    model = NextWordModel(
        processor=process,
        load_existing=False,
        model_name="my_model_name"
    )

    # train the model
    model.train(
        X_train, 
        y_train, 
        epochs = 100, 
        batch_size=128    
    )


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()

    parser.add_argument("sample-factor")
    parser.add_argument("batch-size")
    parser.add_argument("epochs")
    parser.add_argument("model-name")

    args = parser.parse_args()

    sample_factor = args.sample_factor
    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    
    train_model(
        sample_factor,
        model_name,
        epochs,
        batch_size
    )