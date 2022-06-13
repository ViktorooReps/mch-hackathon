"""
    Суть скрипта:
    сделать инференс используя 
    load_test из inference_logreg.py

    Аргументы argparse:
    * exp-root - папка с экспериментами
    * exp-name - название эксперимента из которого подгружается модель
"""

import os
from os.path import join

import argparse
import joblib 
import numpy as np

import pandas as pd

def load_test():
    num_samples = 200

    string_id = [f'str_{i}' for i in range(num_samples)]
    features = 5 * np.random.randn(num_samples, 30)
    return string_id, features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-root', type=str, default='./experiments')
    parser.add_argument('--exp-name', type=str)

    args = parser.parse_args()
    exp_path = join(args.exp_root, args.exp_name)

    model = joblib.load(join(exp_path, 'logreg_model.pkl'))
        
    ids, features = load_test()
    targets = model.predict(features)
    df = pd.DataFrame({
        'id' : ids,
        'target' : targets 
    })
    df.to_csv(join(exp_path, 'test_preds.csv'), index=False)
