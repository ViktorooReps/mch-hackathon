"""
    Суть скрипта: натренировать модель 
    LogisticRegression + узнать ROC-AUC используя
    load_train_val из optuna_tuner.py

    Аргументы argparse:
    * exp-root - папка с экспериментами
    * exp-name - название эксперимента

    Для работы со своими данными нужно реализовать 
    ```load_train_val``` метод.
"""

import os
from os.path import join
import shutil

import argparse

import joblib
import json

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def load_train_val():
    """
        Для тренировки нужно заимплементить этот метод, - 
        он должен возвращать X_train, y_Train, X_valid, y_valid
    """
    X, y = load_breast_cancer(return_X_y=True)
    border = 300

    X_train, X_valid = X[:border], X[border:]
    y_train, y_valid = y[:border], y[border:]

    return X_train, y_train, X_valid, y_valid
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-root', type=str, default='./experiments')
    parser.add_argument('--exp-name', type=str)

    args = parser.parse_args()
    
    exp_path = join(args.exp_root, args.exp_name) 
    os.makedirs(exp_path, exist_ok=False)

    X_train, y_train, X_valid, y_valid = load_train_val()

    best_model = LogisticRegression(solver='liblinear', max_iter=2000).fit(X_train, y_train)
    score = metrics.roc_auc_score(y_valid, best_model.predict_proba(X_valid)[:, 1])
    
    best_model_config = {
        'score' : score
    }
    with open(join(exp_path, 'training_dump.json'), 'w') as outfile:
        json.dump(best_model_config, outfile, indent=4)
    
    joblib.dump(best_model, join(exp_path, 'logreg_model.pkl'))
    exp_source_dir = join(exp_path, 'source_code')
    os.makedirs(exp_source_dir)
    
    for file_name in ['train_logreg.py', 'train_logreg.sh', 'inference_logreg.py']: 
        shutil.copyfile(file_name, join(exp_source_dir, file_name))
    