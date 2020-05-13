import pandas as pd
import numpy as np
import scipy.special
import math
import gc
import copy
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import os

#Import Data
from ReadAndCleanData import train_csv, test_csv, submission_csv, structures_csv



###########################################################
## We build a Pipeline for Adding Features to our Datasets#
###########################################################


#Import Features from  FeatureFunctions.py
import FeatureFunctions as FF

## Assemble total dataset with features from original data set
def build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=10):
    base, structures = FF.build_type_dataframes(some_csv, structures_csv, coupling_type)
    base = FF.add_coordinates(base, structures, 0)
    base = FF.add_coordinates(base, structures, 1)

    base = base.drop(['atom_0', 'atom_1'], axis=1)
    atoms = base.drop('id', axis=1).copy()
    if 'scalar_coupling_constant' in some_csv:
        atoms = atoms.drop(['scalar_coupling_constant'], axis=1)

    FF.add_center(atoms)
    atoms = atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1)

    atoms = FF.merge_all_atoms(atoms, structures)

    FF.add_distance_to_center(atoms)
    Center_Coordinates = atoms[['x_c', 'y_c', 'z_c']]
    atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)
    atoms.sort_values(['molecule_index', 'atom_index_0', 'atom_index_1', 'd_c'], inplace=True)
    atom_groups = atoms.groupby(['molecule_index', 'atom_index_0', 'atom_index_1'])
    atoms['num'] = atom_groups.cumcount() + 2
    atoms = atoms.drop(['d_c'], axis=1)
    atoms = atoms[atoms['num'] < n_atoms]

    atoms = atoms.set_index(['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()
    atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]
    atoms = atoms.reset_index()

    # downcast back to int8
    for col in atoms.columns:
        if col.startswith('atom_'):
            atoms[col] = atoms[col].fillna(0).astype('int8')

    atoms['molecule_index'] = atoms['molecule_index'].astype('int32')

    full = FF.add_atoms(base, atoms)
    FF.add_distances(full)

    full.sort_values('id', inplace=True)
    full[['x_c', 'y_c', 'z_c']] = Center_Coordinates
    FF.add_distances_to_center(full, n_atoms)
    FF.add_Euler_chars(full)
    FF.add_local_Euler_chars_0(full, 2, n_atoms)
    FF.add_local_Euler_chars_1(full, 2, n_atoms)
    FF.add_2_edge_paths(full, 4, n_atoms)
    return full

## Takes Dataframe with Features and Selects Features of Interest
def take_n_atoms(df, n_atoms, four_start=4):
    labels = []
    for i in range(2, n_atoms):
        label = f'atom_{i}'
        labels.append(label)
    for i in range(n_atoms):
        num = min(i, 4) if i < four_start else 4
        for j in range(num):
            labels.append(f'd_{i}_{j}')
    if 'scalar_coupling_constant' in df:
        labels.append('scalar_coupling_constant')
    labels.append('Euler_Char')
    labels.append('0_local_Euler_char_2')
    labels.append('1_local_Euler_char_2')
    labels.append('num_two_edge_paths_4')
    return df[labels]

########################################################
# Build a GBDT model to predict scalar_coupling values##
########################################################

# Initialize LGB parameters
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.2,
    'num_leaves': 450,
    'min_child_samples': 79,
    'max_depth': 10,
    'subsample_freq': 1,
    'subsample': 0.9,
    'bagging_seed': 11,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 1.0
}

# Builds Train and Test Datasets with Features for one type
def build_x_y_data(some_csv, coupling_type, n_atoms):
    full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)

    df = take_n_atoms(full, n_atoms)
    df = df.fillna(0)

    if 'scalar_coupling_constant' in df:
        X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')
        y_data = df['scalar_coupling_constant'].values.astype('float32')
    else:
        X_data = df.values.astype('float32')
        y_data = None

    return X_data, y_data


## Trains LGB model and predicts for one type
def train_and_predict_for_one_coupling_type(coupling_type, submission, n_atoms, n_folds=5, n_splits=5,
                                            random_state=128):
    print(f'*** Training Model for {coupling_type} ***')

    X_data, y_data = build_x_y_data(train_csv, coupling_type, n_atoms)
    X_test, _ = build_x_y_data(test_csv, coupling_type, n_atoms)
    y_pred = np.zeros(X_test.shape[0], dtype='float32')

    cv_score = 0

    if n_folds > n_splits:
        n_splits = n_folds

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data)):
        if fold >= n_folds:
            break

        X_train, X_val = X_data[train_index], X_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        model = LGBMRegressor(**LGB_PARAMS, n_estimators=6000, n_jobs=-1)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)

        y_val_pred = model.predict(X_val)
        val_score = np.log(mean_absolute_error(y_val, y_val_pred))
        print(f'{coupling_type} Fold {fold}, logMAE: {val_score}')

        cv_score += val_score / n_folds
        y_pred += model.predict(X_test) / n_folds

    submission.loc[test_csv['type'] == coupling_type, 'scalar_coupling_constant'] = y_pred
    return cv_score



## Now we initialize our model building, using parameters specific to each type
model_params = {
    '1JHN': 7,
    '1JHC': 10,
    '2JHH': 9,
    '2JHN': 9,
    '2JHC': 9,
    '3JHH': 9,
    '3JHC': 10,
    '3JHN': 10
}
N_FOLDS = 5
submission = submission_csv.copy()

cv_scores = {}
for coupling_type in model_params.keys():
    cv_score = train_and_predict_for_one_coupling_type(
        coupling_type, submission, n_atoms=model_params[coupling_type], n_folds=N_FOLDS)
    cv_scores[coupling_type] = cv_score

#Check output CV scores
pd.DataFrame({'type': list(cv_scores.keys()), 'cv_score': list(cv_scores.values())})
np.mean(list(cv_scores.values()))

#Output Prediction Results
submission.to_csv('Data/submission.csv')