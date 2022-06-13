"""
Обертка вокруг optuna + LightGBM 
для автоматической тренировки,
черный ящик
"""

import optuna
import lightgbm as lgb

from sklearn import metrics

class AbstractOptunaTuner():
    def __init__(self, param_dict):
        X_train, y_train, X_valid, y_valid = self.load_train_val()
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.n_estimators = param_dict.pop('n_estimators')
        self.time_budget = param_dict.pop('time_budget')
        
    def tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, timeout=self.time_budget)

        # fig = optuna.visualization.plot_optimization_history(study)
        # fig = optuna.visualization.plot_slice(study)
        # fig = optuna.visualization.plot_param_importances(study)
        
        score, model = self.train_lgbm(study.best_params, self.n_estimators)
        return score, model

    def objective(self, trial):
        params = {
            "objective": "binary",
            "metric": "roc-auc",
            # "verbosity": self.verbose,
            "boosting_type": "gbdt",                
            "seed": 42,
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 512),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        }
        score, _ = self.train_lgbm(params, n_estimators=self.n_estimators)
        return score

    def train_lgbm(self, params, n_estimators):
        model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)
        model.fit(self.X_train, self.y_train, 
                eval_set=[(self.X_train, self.y_train), (self.X_valid, self.y_valid)],
                eval_metric='logloss')
        score = metrics.roc_auc_score(self.y_valid, model.predict_proba(self.X_valid)[:, 1])
        return score, model
    
    def load_train_val():
        raise NotImplementedError("Implement me!")
    
    def load_test():
        raise NotImplementedError("Implement me!")