# TUW-ML-3: Next word prediction

This repository contains our (group 50) code to train a LSTM model on the next-word-prediction task. 
All necesarry steps, including preprocesing, training and evlauation are implemented.

## Requirements

As this project was implemented on an Apple Device using a M2 chip, we used the **tensorflow-macos** library.
If you are also a M-chip user you can install all required packages using:

```pip install -r requirements_mac.txt```

otherwise use:

```pip install -r requirements_other_os.txt```

## Data & Processing
As data we used tweets from Donald Trump [https://www.thetrumparchive.com/](https://www.thetrumparchive.com/).
They are loaded using the `DataLoader()` (`src.data_loading.py`) get preprocessed using the `Process()` class (`src.data_preprocessing.py`).

```python
dl = DataLoader()
process = Process(
    dl,
    # the sample factor indicates how much of the data we want to use (can be used to obtain smaller models)
    sample_factor = 0.95    
)
```

## Evaluation
We implemented three different evaluation metrics which are accessible via the `Evaluator()` class which resides in `src.prediction_evaluation`.
For the differences betwen them and how they are computed either look in the source code or in the provided report.

To evaluate a model on (unseen) data use:
```python
# identical word evaluation
accuracy_id =  ev.evaluate(X_test, y_test, eval_type="id")

# lemma evaluation
accuracy_lemma = ev.evaluate(X_test, y_test, eval_type="lemma")

# sentence embedding-based evaluation
accuracy_lemma = ev.evaluate(X_test, y_test, eval_type="embedding")
```

## Reproduction
If you want to reproduce the obtained results, simply run the `ml3.ipynb` notebook, which contains the entire flow & generation of visualisations.


## New Training
If you want to train a new model (on the provided data) you can either do it with python like this:
```python
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
```

You can then load your new model like this:
```python
new_model = NextWordModel(
    processor=process,
    load_existing=True,
    model_name="my_model_name"
)
```

**Note: Training on all data can result in long training times (> 2h)**

