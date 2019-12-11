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
    FF.add_local_valence_0(full, 1.8, n_atoms)
    FF.add_local_valence_1(full, 1.8, n_atoms)
    FF.add_2_edge_paths(full, 1, n_atoms)
    FF.add_2_edge_paths(full, 2, n_atoms)
    FF.add_2_edge_paths(full, 3, n_atoms)
    FF.add_2_edge_paths(full, 4, n_atoms)
    FF.add_local_weight(full,1)
    FF.add_local_weight(full,2)
    FF.add_local_weight(full,3)
    FF.add_local_weight(full,4)
    FF.add_local_weight(full,5)
    return full

## Assemble total dataset with features from original data set
def build_couple_dataframe_with_dists(some_csv, structures_csv, coupling_type, n_atoms=10):
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
    return full