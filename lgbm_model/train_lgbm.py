from abc import abstractmethod
import os
from os.path import join
import shutil

import argparse

import joblib
import json

from optuna_tuner import OptunaTuner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-root', type=str, default='./experiments')
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--time-budget', type=int, default=10)
    parser.add_argument('--num-est', type=str, nargs='*')

    args = parser.parse_args()
    
    exp_path = join(args.exp_root, args.exp_name) 
    os.makedirs(exp_path, exist_ok=False)

    time_budget = args.time_budget
    gridset_nest = [int(x) for x in args.num_est]

    best_score = -1
    best_model_config = None
    best_model = None
    n_est = -1
    
    for n_estimators in gridset_nest: 
        tuner = OptunaTuner({
            'n_estimators' : n_estimators,
            'time_budget' : time_budget
        })
        score, model = tuner.tune()
        if score > best_score:
            best_score = score
            best_model_config = model.get_params()
            best_model = model
            n_est = n_estimators
    
    best_model_config = {
        'n_estimators' : n_est,
        'config' : best_model_config,
        'score' : best_score
    }
    with open(join(exp_path, 'training_dump.json'), 'w') as outfile:
        json.dump(best_model_config, outfile, indent=4)
    joblib.dump(best_model, join(exp_path, 'lgbm_model.pkl'))
    
    exp_source_dir = join(exp_path, 'source_code')
    os.makedirs(exp_source_dir)
    for file_name in ['train_lgbm.py', 'train_lgbm.sh', 'abs_optuna_tuner.py']: 
        shutil.copyfile(file_name, join(exp_source_dir, file_name))
    