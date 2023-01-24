from collections import namedtuple

class Evaluator(object):

    def __init__(self, model):
        self.embedding_model = ""
        self.lemmatizer = ""
        self.model = model

    def _parse_word_input(self, w_true, w_pred):        
        to_list = lambda x: [x] if isinstance(x, str) else x
        w_true_list, w_pred_list = to_list(w_true), to_list(w_pred)

        if len(w_pred_list) != len(w_true_list):
            raise Exception("<w_true> and <w_pred> must have the same length!")

        Pair = namedtuple("Pair", "true pred")
        return [
            Pair(element[0], element[1]) for element in
            list(zip(w_true_list, w_pred_list))
        ]

    def evaluate_lemma(self, w_true, w_pred):
        pairs = self._parse_word_input(w_true, w_pred)
        print(pairs)
        print(pairs[0].true)
        

    
    def evaluate_embedding(self, w_true, w_pred, threshold=0.1):
        pairs = self._parse_word_input(w_true, w_pred)
