# ReadMe
This repo. contains a series of python files (and copy jupyter notebooks) used to analyze
molecular structure data to predict scalar J-coupling constants
between atoms in large molecules.


It contains the following files:

#### ReadAndCleanData.py

Reads molecular structure and label data for use in other files.

#### BuildFeatureDataFrame.py

Constructs pandas dataframe with relevant data in a form more useful for our predictive goals.

#### EDA.py

Exploration of the data, found some interesting relations and potentially useful features.

#### FeatureFunctions.py

Implements functions on dataframe which add features discovered in EDA which seem likely to be useful.

#### DecisionTreeGT

Builds a random forest model using the underlying features from the data as well as the features discovered in EDA.
