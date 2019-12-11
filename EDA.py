import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import scipy.special
import math
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

##Load Data##
from ReadAndCleanData import train_csv, test_csv, submission_csv, structures_csv

#Let's see what kind of data we are dealing with
print("train file has shape:", train_csv.shape)
print("test file has shape:", test_csv.shape)
print("structures file has shape:", structures_csv.shape)

#print(train_csv.head(20))
#print(structures_csv.head(20))

# We aim to analyze and predict the scalar_coupling_constant for arbitrary atoms
# Let's investigate this quantity in the training data

print(train_csv['scalar_coupling_constant'].describe())
sns.distplot(train_csv['scalar_coupling_constant'], color = 'blue')
plt.title("Scalar Coupling Frequency Distribution")
plt.show()

#There are 8 types of bonds occurring in the data
bond_types = train_csv["type"].unique()
print(bond_types)

for bond_type in bond_types:
    print(f"The scalar coupling constant for bond type {bond_type} can be described by")
    print(train_csv[train_csv['type']==bond_type]['scalar_coupling_constant'].describe())
for i, bond_type in enumerate(bond_types):
    plt.subplot(4, 2, i + 1)
    sns.distplot(train_csv[train_csv['type'] == bond_type]['scalar_coupling_constant'], color='blue')
    plt.title(bond_type, x=.8, y=.8)
plt.show()