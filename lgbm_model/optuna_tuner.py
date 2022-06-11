from abs_optuna_tuner import AbstractOptunaTuner
from sklearn.datasets import load_breast_cancer

class OptunaTuner(AbstractOptunaTuner):
    def load_train_val(self):
        X, y = load_breast_cancer(return_X_y=True)
        border = 300

        X_train, X_valid = X[:border], X[border:]
        y_train, y_valid = y[:border], y[border:]

        return X_train, y_train, X_valid, y_valid
    