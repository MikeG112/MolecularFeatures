import pandas as pd
import numpy as np
import scipy.special
import math
import gc
import copy

# Define Constituent Functions

def build_type_dataframes(base, structures, coupling_type):
    base = base[base['type'] == coupling_type].drop('type', axis=1).copy()
    base = base.reset_index()
    base['id'] = base['id'].astype('int32')
    structures = structures[structures['molecule_index'].isin(base['molecule_index'])]
    return base, structures

def add_coordinates(base, structures, index):
    df = pd.merge(base, structures, how='inner',
                  left_on=['molecule_index', f'atom_index_{index}'],
                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
    df = df.rename(columns={
        'atom': f'atom_{index}',
        'x': f'x_{index}',
        'y': f'y_{index}',
        'z': f'z_{index}'
    })
    return df

def add_atoms(base, atoms):
    df = pd.merge(base, atoms, how='inner',
                  on=['molecule_index', 'atom_index_0', 'atom_index_1'])
    return df

def merge_all_atoms(base, structures):
    df = pd.merge(base, structures, how='left',
                  left_on=['molecule_index'],
                  right_on=['molecule_index'])
    df = df[(df.atom_index_0 != df.atom_index) & (df.atom_index_1 != df.atom_index)]
    return df

def add_center(df):
    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))
    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))
    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))

def add_distance_to_center(df):
    df['d_c'] = ((
        (df['x_c'] - df['x'])**np.float32(2) +
        (df['y_c'] - df['y'])**np.float32(2) +
        (df['z_c'] - df['z'])**np.float32(2)
    )**np.float32(0.5))

def add_distance_between(df, suffix1, suffix2):
    df[f'd_{suffix1}_{suffix2}'] = ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) +
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))
def add_distance_to_center_to_train(df,suffix1):
    df[f'd_c_{suffix1}'] = ((
        (df['x_c'] - df[f'x_{suffix1}'])**np.float32(2) +
        (df['y_c'] - df[f'y_{suffix1}'])**np.float32(2) +
        (df['z_c'] - df[f'z_{suffix1}'])**np.float32(2)
    )**np.float32(0.5))


def add_distances(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])

    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_distance_between(df, i, vi)

def add_distances_to_center(df, n_atoms=7):
    for i in range(2, n_atoms):
        add_distance_to_center_to_train(df, i)

def add_n_atoms(base, structures):
    dfs = structures['molecule_index'].value_counts().rename('n_atoms').to_frame()
    return pd.merge(base, dfs, left_on='molecule_index', right_index=True)

#Add Some Features from Scalar Coupling Incidence Graph
# 1) Euler Characteristic of Each Molecule (Assuming we have the right graph)
def add_Euler_chars(df):
    df['N_Nodes'] =np.zeros(len(df))
    df['Euler_Char'] = np.zeros(len(df))
    df['N_Edges'] = df['d_1_0'].groupby(df['molecule_index'], sort = False).transform('size')
    df['N_Nodes'] = df[['atom_index_0','atom_index_1']].groupby(df['molecule_index'], sort = False).transform('max') + 1
    df['Euler_Char'] = df['N_Nodes'] - df['N_Edges']

# 2) Local Euler Characteristic near (within epsilon ball of) atom_0
def add_local_Euler_chars_0(df, epsilon, n_atoms):
    # requires epsilon >=1 to be topological
    # Consider all edges corresponding to distances from  vertices to vertex 0
    edges = df[[f'd_{n}_0' for n in range(1, n_atoms)]].copy(deep=True)
    vertices = edges.copy(deep=True)
    # Construct a truth table of whether each such edge lies within a ball of radius epsilon*d_1_0 of that vertex
    # Equivalently, count the number of vertices which lie in this ball (excluding vertex 0)
    for n in range(1, n_atoms):
        vertices[f'd_{n}_0'] = np.where(vertices[f'd_{n}_0'] <= vertices['d_1_0'].multiply(epsilon), 1, 0)
    # Add each such vertex/edge to the "local" vertex and edge count
    num_vertices_without_zero = vertices.sum(axis=1)
    num_edges_to_zero = num_vertices_without_zero

    # Add the additional edges between other vertices which also lie in this ball to the local edge count
    # The ball is convex, so we must add one edge for every pair of (nonzero) vertices which is contained in the epilson ball
    # There are num_vertices_without_zero choose 2 of these
    num_edges_away_from_zero = scipy.special.comb(num_vertices_without_zero, 2)
    num_edges = num_edges_to_zero + num_edges_away_from_zero
    num_vertices = num_vertices_without_zero + 1

    df[f'0_local_Euler_char_{epsilon}'] = num_vertices - num_edges

# 3) Local Euler Characteristic near (within epsilon ball of) atom_1
def add_local_Euler_chars_1(df, epsilon, n_atoms):
    # requires epsilon >=1 to be topological
    # Consider all edges corresponding to distances from  vertices to vertex 1
    edges = df[[f'd_{n}_1' for n in range(2, n_atoms)]].copy(deep=True)
    edges['d_1_1'] = df['d_1_0']
    vertices = edges.copy(deep=True)
    # Construct a truth table of whether each such edge lies within a ball of radius epsilon*d_1_0 of that vertex
    # Equivalently, count the number of vertices which lie in this ball (excluding vertex 1)
    for n in range(1, n_atoms):
        vertices[f'd_{n}_1'] = np.where(vertices[f'd_{n}_1'] <= vertices['d_1_1'].multiply(epsilon), 1, 0)
    # Add each such vertex/edge to the "local" vertex and edge count
    num_vertices_without_zero = vertices.sum(axis=1)
    num_edges_to_zero = num_vertices_without_zero

    # Add the additional edges between other vertices which also lie in this ball to the local edge count
    # The ball is convex, so we must add one edge for every pair of (nonzero) vertices which is contained in the epilson ball
    # There are num_vertices_without_zero choose 2 of these
    num_edges_away_from_zero = scipy.special.comb(num_vertices_without_zero, 2)
    num_edges = num_edges_to_zero + num_edges_away_from_zero
    num_vertices = num_vertices_without_zero + 1

    df[f'1_local_Euler_char_{epsilon}'] = num_vertices - num_edges

# 4) Number of 2-edge paths between bonding pair (such that path is contained in epsilon ball)
def add_2_edge_paths(df, epsilon, n_atoms):
    # requires epsilon >=1 to be topological
    # Consider all edges corresponding to distances from  vertices to vertex 0
    edges = df[[f'd_{n}_0' for n in range(2, n_atoms)]].copy(deep=True)
    edges['d_1_1'] = df['d_1_0']
    edges[[f'd_{n}_1' for n in range(2, n_atoms)]] = df[[f'd_{n}_1' for n in range(2, n_atoms)]]
    for n in range(2, n_atoms):
        edges[f'd_{n}_0'] = np.where((edges[f'd_{n}_0'] + edges[f'd_{n}_1']) <= edges['d_1_1'].multiply(epsilon), 1, 0)
    edges.drop([f'd_{n}_1' for n in range(1, n_atoms)], axis=1)
    num_two_edge_paths = edges.sum(axis=1)

    df[f'num_two_edge_paths_{epsilon}'] = num_two_edge_paths
    
def add_local_valence_0(df, epsilon, n_atoms):
    #requires epsilon >=1 to be topological
    #Consider all edges corresponding to distances from  vertices to vertex 0
    edges = df[[f'd_{n}_0' for n in range(1,n_atoms)]].copy(deep=True)
    for n in range(1,n_atoms):
        edges[f'd_{n}_0'] = np.where(edges[f'd_{n}_0'] <= edges['d_1_0'].multiply(epsilon), 1, 0)
    edges_to_zero = edges.sum(axis=1) 
    
    df[f'0_local_valence_{epsilon}'] = edges_to_zero
    
def add_local_valence_1(df, epsilon, n_atoms):
    #requires epsilon >=1 to be topological
    #Consider all edges corresponding to distances from  vertices to vertex 0
    edges = df[[f'd_{n}_1' for n in range(2,n_atoms)]].copy(deep=True)
    edges['d_1_1'] = df['d_1_0']
    for n in range(1,n_atoms):
        edges[f'd_{n}_1'] = np.where(edges[f'd_{n}_1'] <= edges['d_1_1'].multiply(epsilon), 1, 0)
    edges_to_one = edges.sum(axis=1) 
    
    df[f'1_local_valence_{epsilon}'] = edges_to_one
    
def add_local_weight(df, num_neighbors):
    # num_neighbors = choose number of nearest neighbors to consider
    # 0 < num_neighbors <= n_atoms
    atoms =df[[f'atom_{i}' for i in range(2,num_neighbors+2)]]
    atomic_weight = atoms.sum(axis=1)
    df[f'local_weight_{num_neighbors}'] = atomic_weight