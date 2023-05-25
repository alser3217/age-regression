import logging as log

import argparse

import numpy as np

import pandas as pd

import torch
import torch.nn as nn

import optuna

from utilities import create_stratified_folds, cv_train_eval, setup_logger, train_cv_catboost, train_cv_lightgbm, train_cv_xgboost

import json

# TO DO: add argparser: no_trials, no_epochs_to_info, params_for_tuning

def objective(trial, dataset, features, target, criterion, model_type, debug, device):

    if model_type == 'catboost':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1),
            'iterations':   trial.suggest_int('iterations', 100, 1200),
        }
        loss = train_cv_catboost(dataset, features, target, params, debug)
    elif model_type == 'lightgbm':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1),
            'n_estimators':   trial.suggest_int('n_estimators', 100, 1200),
        }
        loss = train_cv_lightgbm(dataset, features, target, params, debug)
    elif model_type == 'xgboost':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1),
            'n_estimators':   trial.suggest_int('n_estimators', 100, 1200),
        }
        loss = train_cv_xgboost(dataset, features, target, params, debug)

    else:
        loss = cv_train_eval(dataset, features, target, params, criterion, debug, device)

    return loss

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, help='model_name')
    args = vars(parser.parse_args())
    model = args['model']

    formatter = log.Formatter('%(asctime)s %(levelname)s %(message)s')
    debug = setup_logger('debug', './logs/debug/' + model + '.log', formatter, 'stdout')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    immunology = pd.read_excel('./data/media-1.xlsx', skiprows=1)
    immunology.drop(['index'], axis=1, inplace=True)
    immunology['Sex'].replace(['M', 'F'], [1, 0], inplace=True)
    features = immunology.columns
    target = ['Age']
    target_name = 'Age'

    create_stratified_folds(immunology, target_name, n_s=6, n_grp=5)

    criterion = nn.MSELoss()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, immunology, features, target, criterion, model, debug, device), n_trials=10)

    best_trial = study.best_trial

    formatter = log.Formatter('%(message)s')
    params = setup_logger('debug', './logs/params/' + model + '.log', formatter, 'params')
    params.info(json.dumps(best_trial.params))

if __name__ == "__main__":
    main()
