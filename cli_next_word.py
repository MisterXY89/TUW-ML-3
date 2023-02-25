
import argparse

from src.data_loading import DataLoader
from src.data_processing import Process

from src.next_word_model import NextWordModel
from src.prediction_evaluation import Evaluator


def evaluate_model(model, eval_type, X_test, y_test):
    ev = Evaluator(model)
    if eval_type == "all":
        for etype in ev.eval_types:
            m_acc = ev.evaluate(X_test, y_test, eval_type=etype)
            print(f"> Model accuracy is: {m_acc}% [{etype}]")
    else:
        accuracy = ev.evaluate(X_test, y_test, eval_type=eval_type)
        print(f"> Model accuracy is: {accuracy}%")


def train_model(sample_factor=0.3, model_name="new_model", epochs = 100, batch_size = 128, evaluate="", optimizer="adam", loss="categorical_crossentropy"):
    dl = DataLoader()
    process = Process(
        dl,
        model_name = model_name,
        sample_factor = sample_factor
    )

    # obtain the test and train data
    X_train, X_test, y_train, y_test = process.process(force = True)

    # initalize the model
    model = NextWordModel(
        processor = process,
        load_existing = False,
        model_name = model_name,
        optimizer = optimizer, 
        loss = loss
    )

    # train the model
    model.train(
        X_train, 
        y_train, 
        epochs = epochs, 
        batch_size = batch_size,        
    )

    if evaluate:
        evaluate_model(model, evaluate, X_test, y_test)


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name")
    parser.add_argument("--sample-factor", default = 0.75)
    parser.add_argument("--batch-size", default = 128)
    parser.add_argument("--epochs", default = 100)    
    parser.add_argument("--evaluate", default = "")
    parser.add_argument("--loss", default = "categorical_crossentropy")
    parser.add_argument("--optimizer", default = "adam")

    args = parser.parse_args()

    sample_factor = float(args.sample_factor)
    model_name = args.model_name
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    evaluate = args.evaluate
    loss = args.loss
    optimizer = args.optimizer    
    
    train_model(
        sample_factor = sample_factor,
        model_name = model_name,
        epochs = epochs,
        batch_size = batch_size,
        evaluate = evaluate,
        loss = loss,
        optimizer = optimizer
    )