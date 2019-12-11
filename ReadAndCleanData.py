#################
## Load Datasets#
#################
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print("Loading Data...")

# Load Train Data
train_dtypes = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
train_csv = pd.read_csv('Data/train.csv', index_col='id', dtype=train_dtypes)
train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
train_csv = train_csv[
    ['molecule_name', 'molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]

# Load Test Data
test_csv = pd.read_csv('Data/test.csv', index_col='id', dtype=train_dtypes)
test_csv['molecule_index'] = test_csv['molecule_name'].str.replace('dsgdb9nsd_', '').astype('int32')
test_csv = test_csv[['molecule_name', 'molecule_index', 'atom_index_0', 'atom_index_1', 'type']]

# Load Sample Submission
submission_csv = pd.read_csv('Data/sample_submission.csv', index_col='id')

# Load Structures Data
structures_dtypes = {
    'molecule_name': 'category',
    'atom_index': 'int8',
    'atom': 'category',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32'
}
# Will use atomic numbers to recode atomic names
ATOMIC_NUMBERS = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9
}
structures_csv = pd.read_csv('Data/structures.csv', dtype=structures_dtypes)
structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
structures_csv = structures_csv[['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]
structures_csv['atom'] = structures_csv['atom'].replace(ATOMIC_NUMBERS).astype('int8')

print("Data Captured")