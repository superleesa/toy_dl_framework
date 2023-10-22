import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    @abstractmethod
    def get_score(self, X: np.ndarray, y: np.ndarray) -> int:
        ...


class Accuracy(Metric):
    def get_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """y_true should be an output from softmax (should have 2 dimensions)"""
        correct_count = (y_pred.argmax(axis=1) == y_true).sum()
        score = correct_count / len(y_true)
        return score

class MAE(Metric):
    def get_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """y_true should be an output from softmax (should have 2 dimensions)"""
        score = np.abs(y_true - y_pred)
        return score

