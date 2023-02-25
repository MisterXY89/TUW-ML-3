import numpy as np
from collections import namedtuple

import nltk
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer, util

import tensorflow as tf

nltk.download('wordnet')

class Evaluator(object):

    def __init__(self, model):
        self.embedding_model = ""
        self.lemmatizer = ""
        self.model = model
        self.lemmatizer = WordNetLemmatizer()
        self.pred = None
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.eval_types = ["id", "lemma", "embedding"]

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
        test_words = self._get_words(
            X_test            
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

        self.errors = errors        
        return preds


    def _acc(self, pred_eval, report = True):
        acc = round((sum(pred_eval)/(len(pred_eval))) * 100, 2)
        if report:          
            print(f"Accuracy: {acc}%")
        return acc       

    def evaluate(self, X_test, y_test, eval_type = "id", force = False, threshold = 0.7):
        

        assert eval_type in self.eval_types, f"available evaluation types include: {self.eval_types}"            

        if (not force and not self.pred) or force or not self.pred:
            self.pred = w_pred = self._get_preds(X_test)            
        else:
            w_pred = self.pred

        pairs = self._parse_word_input(y_test, w_pred)

        if eval_type == "id":
            return self.evaluate_identical(pairs)
        if eval_type == "lemma":
            return self.evaluate_lemma(pairs)
        if eval_type == "embedding":
            return self.evaluate_embedding(X_test, pairs, threshold=threshold)
        

    def evaluate_identical(self, pairs):
        pred_eval = [1 if pair.true == pair.pred else 0 for pair in pairs]
        return self._acc(pred_eval, report=True)

    def evaluate_lemma(self, pairs):
        """
        see https://www.geeksforgeeks.org/python-lemmatization-with-nltk/amp/
        """        
        lemma_compare = lambda x: self.lemmatizer.lemmatize(x.true) == self.lemmatizer.lemmatize(x.pred)
        pred_eval = [1 if lemma_compare(pair) else 0 for pair in pairs]
        return self._acc(pred_eval, report=True)

    def __build_sentence(self, x_test_token, pair_value):
        x_test_word = self._get_words([x_test_token])[0]
        return f"{x_test_word} {pair_value}"
        
    def evaluate_embedding(self, X_test, pairs, threshold):        
        """
        see https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        and https://studymachinelearning.com/cosine-similarity-text-similarity-metric/
        and https://www.learndatasci.com/glossary/cosine-similarity/
        """
        sentences_true = [self.__build_sentence(X_test[idx], pair.true) for idx, pair in enumerate(pairs)]
        sentences_pred = [self.__build_sentence(X_test[idx], pair.pred) for idx, pair in enumerate(pairs)]

        # print(sentences_true)
        # print(sentences_pred)

        embeddings_true = self.sbert_model.encode(sentences_true, convert_to_tensor=True)
        embeddings_pred = self.sbert_model.encode(sentences_pred, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(embeddings_true, embeddings_pred)
        pair_scores = tf.linalg.tensor_diag_part(cosine_scores)

        pred_eval = [1 if score > threshold else 0 for score in pair_scores]

        return self._acc(pred_eval, report=True), pair_scores
