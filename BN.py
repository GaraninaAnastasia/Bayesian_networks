# import libraries
# ! pip install causalnex
# ! pip install pgmpy
# ! pip install bnlearn
# ! pip install numpy
# !pip install bnlearn

import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
import pgmpy.estimators as ests
import os
import random
import warnings
import networkx as nx
import statsmodels
from scipy.io import arff
from scipy.signal._signaltools import _centered
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.metrics import structure_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import bnlearn as bn
from sklearn.preprocessing import LabelEncoder


# Learn the DAG using Chow-liu
model_cl      = bn.structure_learning.fit(struct_data, methodtype='cl', root_node='absences', scoretype= 'k2')
# Plot detected DAG
G = bn.plot(model_cl)

# Learn the DAG using Tree-augmented Naive Bayes
model_tan = bn.structure_learning.fit(struct_data, methodtype='tan', class_node='absences', scoretype= 'k2')
# Plot detected DAG
G = bn.plot(model_tan)


# Read data
# student_mat_data = pd.read_csv("student-mat.csv")
# data = student_mat_data

student_por_data = pd.read_csv("student-por.csv")
data = student_por_data

# data preprocessing
data.notnull()

struct_data = data
discretised_data = struct_data.copy()

data_vals = {col: struct_data[col].unique() for col in struct_data.columns}

failures_map = {v: 'no-failure' if v == [0]
                else 'have-failure' for v in data_vals['failures']}
studytime_map = {v: 'short-studytime' if v in [1,2]
                 else 'long-studytime' for v in data_vals['studytime']}

discretised_data["failures"] = discretised_data["failures"].map(failures_map)
discretised_data["studytime"] = discretised_data["studytime"].map(studytime_map)

from causalnex.discretiser import Discretiser

discretised_data["absences"] = Discretiser(method="fixed",
                          numeric_split_points=[1, 10]).transform(discretised_data["absences"].values)
discretised_data["G1"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G1"].values)
discretised_data["G2"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G2"].values)
discretised_data["G3"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G3"].values)

absences_map = {0: "No-absence", 1: "Low-absence", 2: "High-absence"}

G1_map = {0: "Fail", 1: "Pass"}
G2_map = {0: "Fail", 1: "Pass"}
G3_map = {0: "Fail", 1: "Pass"}

discretised_data["absences"] = discretised_data["absences"].map(absences_map)
discretised_data["G1"] = discretised_data["G1"].map(G1_map)
discretised_data["G2"] = discretised_data["G2"].map(G2_map)
discretised_data["G3"] = discretised_data["G3"].map(G3_map)

G_ = "G1"
struct_data = discretised_data.copy()
non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in non_numeric_columns:
    struct_data[col] = le.fit_transform(struct_data[col])

drop_col = ['school','sex','age','Mjob', 'Fjob','reason','guardian']
struct_data = struct_data.drop(columns=drop_col)

from sklearn.model_selection import train_test_split
train, test = train_test_split(struct_data, train_size=0.9, test_size=0.1, random_state=7)

# Learn the DAG using Chow-liu
model_cl = bn.structure_learning.fit(train, methodtype='cl', root_node='absences', scoretype= 'k2')
# Plot detected DAG
G = bn.plot(model_cl)
# Parameter learning
model = bn.parameter_learning.fit(model_cl, train, verbose=3);

# Learn the DAG using Tree-augmented Naive Bayes
# model_tan = bn.structure_learning.fit(struct_data, methodtype='tan', class_node='absences', scoretype= 'k2')
# # Plot detected DAG
# G = bn.plot(model_tan)
# # Parameter learning
# model = bn.parameter_learning.fit(model_tan, train, verbose=3);

Pout = bn.predict(model, test, variables=[G_])
# print(Pout)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = bn.predict(model, test, variables=[G_])[G_]
print("")
print(confusion_matrix(test[G_], predictions))
print(classification_report(test[G_], predictions))



# Linear Regression
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
# Extract feature columns
feature_cols = list(struct_data.columns[:-3])
#feature_cols

# Extract target column 'passed'
#target_col = student_data.columns[-3:]
target_col_G1 = struct_data.columns[-3]
target_col_G2 = struct_data.columns[-2]
target_col_G3 = struct_data.columns[-1]

target_col = target_col_G3
#target_col

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = struct_data[feature_cols]
y_all = struct_data[target_col]

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=0)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("")
print(confusion_matrix(np.array(y_test), np.rint(y_pred)))
print(classification_report(np.array(y_test), np.rint(y_pred)))