import copy
import logging as log
from typing import Tuple

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

class ImmunologyDataset(Dataset):
    def __init__(self, tensor_X, tensor_y, device):
        self.inputs = tensor_X.to(device)
        self.targets = tensor_y.to(device)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size=10, output_size=1):
        super(MyNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 172)
        self.fc2 = nn.Linear(172, 164)
        self.fc3 = nn.Linear(164, 140)
        self.fc4 = nn.Linear(140, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, output_size)

        self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        
        self.batch_normalization = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))

        return x
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None
        self.best_epoch = None

    def __call__(self, epoch, val_loss, logger, model):
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f'Early stopping triggered after {epoch} epochs.')
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        return False
    
def setup_logger(name, path, formatter, type, level=log.INFO):
    logger = log.getLogger(name)
    logger.setLevel(level)

    handler = log.FileHandler(path, mode='w')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if type == 'stdout':
        handler = log.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def create_stratified_folds(dataset, target_name, n_s=5, n_grp=None):

    dataset['Fold'] = -1
    skf = StratifiedKFold(n_splits=n_s, random_state=43, shuffle=True)
    dataset['grp'], bins_range = pd.cut(dataset[target_name], n_grp, labels=False, retbins=True)
    target = dataset.grp
    
    for fold_no, (_, v) in enumerate(skf.split(target, target)):
        dataset.loc[v, 'Fold'] = fold_no
    return dataset, bins_range

def train(rgs, train_dataloader, val_dataloader, test_dataloader, params, criterion, logger, eval=False): # FIX TRAIN/EVAL LOSSES COMPUTATION!!!
    logger.info('---Started training---')

    early_stopping = EarlyStopping(patience=200)

    num_epochs = params['epochs']
    optimizer = torch.optim.Adam(rgs.parameters(), lr=params['learning_rate'])

    for epoch in range(num_epochs):
        train_loss, val_loss, test_loss = 0, 0, 0
        rgs.train()
        for data in train_dataloader:
            img, labels = data
            optimizer.zero_grad()
            output = rgs(img)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if eval:
            with torch.no_grad():
                rgs.eval()
                for data in val_dataloader:
                    img, labels = data
                    output = rgs(img)
                    loss_eval = criterion(output, labels)
                    val_loss += loss_eval.item()
            if epoch % 50 == 0:
                logger.info(f'epoch [{epoch}/{num_epochs}], train_loss: {train_loss}, val_loss: {val_loss}')
        else:
            with torch.no_grad():
                rgs.eval()
                for data in test_dataloader:
                    img, labels = data
                    output = rgs(img)
                    loss_test = criterion(output, labels)
                    test_loss += loss_test.item()
            if epoch % 50 == 0:
                logger.info(f'epoch [{epoch}/{num_epochs}], train_loss: {train_loss}, test_loss: {test_loss}')
        if early_stopping(epoch, test_loss, logger, rgs):
            break
    logger.info('---Finished training---')

    return early_stopping.best_model

def cv_train_eval(dataset, features, target, params, criterion, logger, device):

    y_pred_val_list = []
    y_true_val_list = []

    mean_maes = []

    num_epochs = params['epochs']
    train_eval_folds = [i for i in range(dataset['Fold'].max())]
    for i in train_eval_folds:

        rgs = MyNeuralNetwork(input_size=len(features)).to(device)
        optimizer = torch.optim.Adam(rgs.parameters(), lr=params['learning_rate'])

        logger.info('\n')
        logger.info(f'---Perfoming cross-validation training/evaluating. Evaluating fold {i}---\n')
        X_train = dataset[features].loc[dataset['Fold'] != i]
        y_train = dataset[target].loc[dataset['Fold'] != i]

        X_val = dataset[features].loc[dataset['Fold'] == i]
        y_val = dataset[target].loc[dataset['Fold'] == i]

        immunology_train_dataset = ImmunologyDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32), device)
        immunology_val_dataset = ImmunologyDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32), device)

        train_dataloader = DataLoader(immunology_train_dataset, batch_size=128, shuffle=True)
        val_dataloader = DataLoader(immunology_val_dataset, batch_size=16, shuffle=True)
        
        for epoch in range(num_epochs):
            train_loss, val_loss = 0, 0
            rgs.train()
            for data in train_dataloader:
                img, labels = data
                optimizer.zero_grad()
                output = rgs(img)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            with torch.no_grad():
                rgs.eval()
                for data in val_dataloader:
                    img, labels = data
                    output = rgs(img)
                    loss_eval = criterion(output, labels)
                    val_loss += loss_eval.item()
                    for prediction in output:
                        prediction = prediction.numpy()
                        y_pred_val_list.append(prediction)
                    for label in labels:
                        label = label.numpy()
                        y_true_val_list.append(label)
            if epoch % 50 == 0:         
                logger.info(f'epoch [{epoch}/{num_epochs}], train_loss: {train_loss}, val_loss: {val_loss}')
        mean_maes.append(mean_absolute_error(y_true_val_list, y_pred_val_list))
        logger.info(f'validation fold: {i}, mae: {mean_maes[i]}')                        
    logger.info('\n')
    logger.info('---Finished training---')
    logger.info(f'Mean of mae over all folds: {np.mean(mean_maes)}')
    return np.mean(mean_maes)

def test(rgs, test_dataloader):
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            y_pred = rgs(inputs)
            for prediction in y_pred:
                prediction = prediction.numpy()
                y_pred_list.append(prediction)
            for label in labels:
                label = label.numpy()
                y_true_list.append(label)

    return mean_absolute_error(y_true_list, y_pred_list)

def train_cv_catboost(dataset, features, target, params, logger=None):

    mean_maes = []

    train_eval_folds = [i for i in range(dataset['Fold'].max())]
    for i in train_eval_folds:

        rgs = CatBoostRegressor(**params, loss_function='RMSE', eval_metric='MAE', silent=True)

        if logger is not None:
            logger.info('\n')
            logger.info(f'---Perfoming cross-validation training/evaluating. Evaluating fold {i}---\n')
        X_train = dataset[features].loc[(dataset['Fold'] != i) & (dataset['Fold'] != dataset['Fold'].max())]
        y_train = dataset[target].loc[(dataset['Fold'] != i) & (dataset['Fold'] != dataset['Fold'].max())]

        X_val = dataset[features].loc[dataset['Fold'] == i]
        y_val = dataset[target].loc[dataset['Fold'] == i]

        rgs.fit(X_train, y_train, eval_set=[(X_val, y_val)], use_best_model=True)

        results = rgs.best_score_
        mean_maes.append(results['validation']['MAE'])

        logger.info(f'validation fold: {i}, mae: {mean_maes[i]}')   
    if logger is not None:                     
        logger.info('\n')
        logger.info('---Finished training---')
        logger.info(f'Mean of mae over all folds: {np.mean(mean_maes)}')

    return np.mean(mean_maes)

def train_cv_lightgbm(dataset, features, target, params, logger):

    mean_maes = []

    train_eval_folds = [i for i in range(dataset['Fold'].max())]
    for i in train_eval_folds:

        rgs = LGBMRegressor(**params)

        logger.info('\n')
        logger.info(f'---Perfoming cross-validation training/evaluating. Evaluating fold {i}---\n')
        X_train = dataset[features].loc[dataset['Fold'] != i & (dataset['Fold'] != dataset['Fold'].max())]
        y_train = dataset[target].loc[dataset['Fold'] != i & (dataset['Fold'] != dataset['Fold'].max())]

        X_val = dataset[features].loc[dataset['Fold'] == i]
        y_val = dataset[target].loc[dataset['Fold'] == i]

        rgs.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric='MAE')

        results = rgs.best_score_
        logger.info(results)

        mean_maes.append(results['valid_0']['l1'])

        logger.info(f'validation fold: {i}, mae: {mean_maes[i]}')                        
    logger.info('\n')
    logger.info('---Finished training---')
    logger.info(f'Mean of mae over all folds: {np.mean(mean_maes)}')

    return np.mean(mean_maes)

def train_cv_xgboost(dataset, features, target, params, logger):

    mean_maes = []

    train_eval_folds = [i for i in range(dataset['Fold'].max())]
    for i in train_eval_folds:

        rgs = XGBRegressor(**params, eval_metric=mean_absolute_error, early_stopping_rounds=100)

        logger.info('\n')
        logger.info(f'---Perfoming cross-validation training/evaluating. Evaluating fold {i}---\n')
        X_train = dataset[features].loc[dataset['Fold'] != i & (dataset['Fold'] != dataset['Fold'].max())]
        y_train = dataset[target].loc[dataset['Fold'] != i & (dataset['Fold'] != dataset['Fold'].max())]

        X_val = dataset[features].loc[dataset['Fold'] == i]
        y_val = dataset[target].loc[dataset['Fold'] == i]

        rgs.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        results = rgs.best_score
        logger.info(results)

        mean_maes.append(results)

        logger.info(f'validation fold: {i}, mae: {mean_maes[i]}')                        
    logger.info('\n')
    logger.info('---Finished training---')
    logger.info(f'Mean of mae over all folds: {np.mean(mean_maes)}')

    return np.mean(mean_maes)

def train_test_fit_like_models(rgs, train_X, train_y, test_X, test_y, logger=None):

    rgs.fit(train_X, train_y)   

    predictions_train = rgs.predict(train_X)
    predictions = rgs.predict(test_X)
    
    if logger is not None:
        logger.info(mean_absolute_error(test_y, predictions))

    return predictions_train, predictions