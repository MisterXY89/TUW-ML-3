# TUW-ML-3: Next word prediction

This repository contains our (group 50) code to train a LSTM model on the next-word-prediction task. 
All necesarry steps, including preprocesing, training and evlauation are implemented.

## Requirements & Building
As this project was implemented on an Apple Device using a M2 chip, using a **Python 3.9.12** virtual environment.
We used the **tensorflow-macos** library. If you are also a M-chip user you can install all required packages using:

```pip install -r requirements_mac.txt```

otherwise use:

```pip install -r requirements_other_os.txt```

Besides this, there are no steps necessary to build/use the project/model.

## Data & Processing
As data we used tweets from Donald Trump [https://www.thetrumparchive.com/](https://www.thetrumparchive.com/).
They are loaded using the `DataLoader()` (`src.data_loading.py`) get preprocessed using the `Process()` class (`src.data_preprocessing.py`).

```python
from src.data_loading import DataLoader
from src.data_processing import Process

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
from src.prediction_evaluation import Evaluator

ev = Evaluator(model)

# identical word evaluation
accuracy_id =  ev.evaluate(X_test, y_test, eval_type="id")

# lemma evaluation
accuracy_lemma = ev.evaluate(X_test, y_test, eval_type="lemma")

# sentence embedding-based evaluation
accuracy_lemma = ev.evaluate(X_test, y_test, eval_type="embedding")
```

## Reproduction
If you want to reproduce the obtained results, simply run the `ml3.ipynb` notebook, which contains the entire flow & generation of visualisations.


## Training of a new model
If you want to train a new model (on the provided data) you can do it like this:
```python
from src.data_loading import DataLoader
from src.data_processing import Process

from src.next_word_model import NextWordModel

from src.prediction_evaluation import Evaluator

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
```

or via the comandline using the `train_new_model.py` file:
```
$ python cli_train_new_model.py <args>
```
with the following arguments:
```
sample-factor       (sampled) fraction of the data which is used
batch-size          batch size used in the training process
epochs              epochs used in the training process
model-name          used to load the model later on
evaluate            name of the method you want to use for evaluation (id, lemma, embedding)
optimizer           name of the TF optimizer
loss                name of the TF loss-function
```    

**Example:**
```
$ python cli_next_word.py --sample-factor 0.1 --model-name test --evaluate lemma --epochs 10    
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

