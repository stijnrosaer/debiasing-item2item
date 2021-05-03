import numpy as np

class Metric():
    pass

class BinaryMetric(Metric):
    def __init__(self, k=-1):
        self._k = k

    def name(self):
        return self.__class__.__name__.replace("K", str(self._k))

    def score(self, y_true, y_score, candidates):
        k = self._k if self._k > 0 else len(y_true)

        n_pos = len(y_true)

        tmp = np.argsort(y_score)[::-1]
        order = [candidates[i] for i in tmp]
        y_pred_labels = [1 if i in y_true else 0 for i in order[:k]]

        n_relevant = np.sum(y_pred_labels)
        return self._score(n_pos, n_relevant, k, y_pred_labels)

class RecallAtK(BinaryMetric):
    def _score(self, n_pos, n_relevant, k, y_pred_labels, y_true=[]):
        if n_pos == 0:
            return 0.0  # if there are no relevant items, return 0
        else:
            return float(n_relevant) / n_pos