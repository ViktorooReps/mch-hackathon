import pickle

import numpy as np

from abs_optuna_tuner import AbstractOptunaTuner
from sklearn.datasets import load_breast_cancer


class OptunaTuner(AbstractOptunaTuner):

    def load_test(self):
        pass

    def load_train_val(self):
        """
            Для тренировки нужно заимплементить этот метод, - 
            он должен возвращать X_train, y_Train, X_valid, y_valid
        """
        with open('cached.pkl', 'rb') as f:
            X, y = pickle.load(f)

        border = int(0.2 * len(y))

        X_train, X_valid = X[:border], X[border:]
        y_train, y_valid = y[:border], y[border:]

        return X_train, y_train, X_valid, y_valid
    