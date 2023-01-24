

from data_loading import DataLoader
from data_processing import Process

from next_word_model import NextWordModel

from prediction_evaluation import Evaluator

if __name__ == "__main__":
    dl = DataLoader()
    process = Process(dl)
    data = process.process(force=False)
    print(data)

    model = NextWordModel(
        sequence_length = process.sequence_length,
        vocab_size = process.vocab_size

    )

    model.train(data["X"], data["y"])

    # ev = Evaluator(model)
    # ev.evaluate_lemma("w13e", "e2")