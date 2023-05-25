import logging as log

import argparse

import pandas as pd

import torch
import torch.nn as nn

from utilities import create_stratified_folds, MyNeuralNetwork, ImmunologyDataset, train, test, setup_logger, train_test_fit_like_models
from torch.utils.data import DataLoader

import json

from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

# def main():

#     formatter = log.Formatter('%(asctime)s %(levelname)s %(message)s')
#     result = setup_logger('result', './logs/results/nn_result.log', formatter, 'stdout')

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     immunology = pd.read_excel('./data/media-1.xlsx', skiprows=1)
#     features = ['FLT3L', 'IL1RA', 'IL6', 'IL25', 'CCL2', 'CSF1', 'CCL22', 'CXCL9', 'CCL4', 'PDGFA']
#     target = ['Age']
#     target_name = 'Age'

#     create_stratified_folds(immunology, target_name, n_s=6, n_grp=5)

#     with open('./logs/params/nn_params.log', 'r') as f:
#         data = f.read()

#     params = json.loads(data)
    
#     rgs = MyNeuralNetwork(len(features)).to(device)
#     criterion = nn.MSELoss()

#     X_train = immunology[features].loc[(immunology['Fold'] != immunology['Fold'].max())]
#     y_train = immunology[target].loc[(immunology['Fold'] != immunology['Fold'].max())]

#     X_test = immunology[features].loc[immunology['Fold'] == immunology['Fold'].max()]
#     y_test = immunology[target].loc[immunology['Fold'] == immunology['Fold'].max()]

#     immunology_train_dataset = ImmunologyDataset(torch.tensor(X_train.values, dtype=torch.float32),
#                                                   torch.tensor(y_train.values, dtype=torch.float32), device)
#     immunology_test_dataset = ImmunologyDataset(torch.tensor(X_test.values, dtype=torch.float32),
#                                                  torch.tensor(y_test.values, dtype=torch.float32), device)
    
#     train_dataloader = DataLoader(immunology_train_dataset, batch_size=64, shuffle=True)
#     test_dataloader = DataLoader(immunology_test_dataset, batch_size=8, shuffle=True)

#     best_rgs = MyNeuralNetwork(len(features)).to(device)
#     best_rgs.load_state_dict(train(rgs, train_dataloader, None, test_dataloader, params, criterion, result))

#     result.info(f'Final accuracy: {test(best_rgs, test_dataloader)}')

def main():

    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('--model', type=str, help='model_name')
    args = vars(parser.parse_args())
    model = args['model']

    formatter = log.Formatter('%(asctime)s %(levelname)s %(message)s')
    result = setup_logger('result', './logs/results/' + model + '.log', formatter, 'stdout')

    immunology = pd.read_excel('./data/media-1.xlsx', skiprows=1)
    immunology.drop(['index'], axis=1, inplace=True)
    immunology['Sex'].replace(['M', 'F'], [1, 0], inplace=True)
    features = immunology.columns
    target = ['Age']
    target_name = 'Age'

    create_stratified_folds(immunology, target_name, n_s=6, n_grp=5)

    with open('./logs/params/' + model + '.log', 'r') as f:
        data = f.read()

    params = json.loads(data)

    X_train = immunology[features].loc[(immunology['Fold'] != immunology['Fold'].max())]
    y_train = immunology[target].loc[(immunology['Fold'] != immunology['Fold'].max())]

    X_test = immunology[features].loc[immunology['Fold'] == immunology['Fold'].max()]
    y_test = immunology[target].loc[immunology['Fold'] == immunology['Fold'].max()]

    if model == 'catboost':
        rgs = CatBoostRegressor(**params, early_stopping_rounds=100, silent=True)
    elif model == 'lightgbm':
        rgs = LGBMRegressor(**params)
    elif model == 'xgboost':
        rgs = XGBRegressor(**params)

    train_test_fit_like_models(rgs, X_train, y_train, X_test, y_test, result)


if __name__ == "__main__":
    main()
