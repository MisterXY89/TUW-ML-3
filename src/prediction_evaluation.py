import numpy as np
from collections import namedtuple

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

class Evaluator(object):

    def __init__(self, model):
        self.embedding_model = ""
        self.lemmatizer = ""
        self.model = model
        self.lemmatizer = WordNetLemmatizer()
        self.pred = None

    def _extract_tokens(self, x):
        # return [*map(lambda y: np.argwhere(y == np.amax(y)).flatten().tolist(), x)]
        return [*map(lambda y: y.argmax(), x)]

    def _get_words(self, x):
        return [*map(lambda y: [key for key, value in self.model.processor.tokenizer.word_index.items() if value == y][0], x)]

    def _parse_word_input(self, w_true, w_pred):        
        to_list = lambda x: [x] if isinstance(x, str) else x

        w_true_list, w_pred_list = to_list(w_true), to_list(w_pred)

        w_true_list = self._extract_tokens(w_true_list)
        w_true_list = self._get_words(w_true_list)

        if len(w_pred_list) != len(w_true_list):
            raise Exception("<w_true> and <w_pred> must have the same length!")

        Pair = namedtuple("Pair", "true pred")
        return [
            Pair(element[0], element[1]) for element in
            list(zip(w_true_list, w_pred_list))
        ]

    def _get_preds(self, X_test):       
        # print(X_test) 
        test_words = self._get_words(
            X_test
            # self._extract_tokens(X_test)
        )

        errors = 0
        preds = []
        for word in test_words:
            try:
                pred = self.model.predict(word)
                preds.append(pred)
            except Exception as e:
                preds.append("")
                errors += 1
        
        print("ERRORS: ", errors)
        return preds

    def _acc(self, pred_eval):
        return (sum(pred_eval)/len(pred_eval)) * 100


    def evaluate(self, X_test, y_test, eval_type = "id", force = False, threshold = 0.1):
        eval_types = ["id", "lemma", "embedding"]

        if not eval_type in eval_types:
            print(f"available evaluation types include: {eval_types}")
            return 0

        if (not force and not self.pred) or force or not self.pred:
            self.pred = w_pred = self._get_preds(X_test)            
        else:
            w_pred = self.pred

        pairs = self._parse_word_input(y_test, w_pred)

        if eval_type == "id":
            self.evaluate_identical(pairs)
        if eval_type == "lemma":
            self.evaluate_lemma(pairs)
        if eval_type == "embedding":
            self.evaluate_embedding(pairs, threshold=threshold)
        

    def evaluate_identical(self, pairs):
        pred_eval = [1 if pair.true == pair.pred else 0 for pair in pairs]
        print("Accuracy ID: ", self._acc(pred_eval))

    def evaluate_lemma(self, pairs):        
        lemma_compare = lambda x: self.lemmatizer.lemmatize(x.true) == self.lemmatizer.lemmatize(x.pred)
        pred_eval = [1 if lemma_compare(pair) else 0 for pair in pairs]
        print("Accuracy Lemma: ", self._acc(pred_eval))
        
    
    def evaluate_embedding(self, pairs, threshold):
        pass
