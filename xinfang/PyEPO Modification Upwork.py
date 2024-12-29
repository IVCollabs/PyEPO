# -*- coding: utf-8 -*-
"""
Spyder Editor

This version adds the base case; which is to predcit then optimize. 
Also save x and c.


"""
import numpy as np
import pandas as pd
import pyepo
import random
import torch

from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from classes_and_methods import LinearRegression, trainModel, visLearningCurve
import os

# Parameters
NUM_DATA = 10   # number of training instances
NUM_FEAT = 5    # number of features x (to predict c)
NUM_NODE = 4    # number of nodes in the network
NUM_EPOCHS = 10 # number of epochs for training
BATCH_SIZE = 10 # batch size for training
OUTPUT_PATH = "outputs/"

# Verifies if outputs path folder exist,
# if not, creates one.
os.makedirs(OUTPUT_PATH, exist_ok=True)

# set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# number of edges, converted to integer
NUM_EDGE = int(NUM_NODE * (NUM_NODE - 1) / 2)  

# A PyEPO function to generate synthetic data and features for travelling salesman
# See https://khalil-research.github.io/PyEPO/build/html/content/examples/data.html

# Note that x is of size (1000, 5) - 1000 samples/data points and 5 feature variables
# c is of size (1000, 190) - c is the target variable, which represents travel time on an edage
# For NUM_NODEs = 20, the graph is typically a complete graph (meaning every node is connected 
# to every other node). E is given by the formula:n*(n - 1)/2 = 20*19/2 = 190
# So, the model uses 5 feature variables to predict 190 target variables. 

# Generate x, c. 
x, c = pyepo.data.tsp.genData(NUM_DATA, NUM_FEAT, NUM_NODE, deg=4, noise_width=0, seed=135)

# build optimization model by calling the single travelling salesman model

"""
Build a new class within grb.Single TSP.py based on mSkilledmTSP.py. 
So, I can build optmodel as follows. 
optmodel = pyepo.model.grb.MStspMTZModel(NUM_NODE) 
# optmodel.setObj(c[0]) # set objective function
# sol, obj = optmodel.solve() 
"""

optmodel = pyepo.model.grb.tspMTZModel(NUM_NODE)

# Below solves a single instance given c[0]
# optmodel.setObj(c[0]) # set objective function
# sol, obj = optmodel.solve() 

# split train test data
# x is feats (feature variables) and c is cost (target variable)

x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=0.2, random_state=42)

# get optDataset
# build dataset: This class is Torch Dataset for optimization problems.
# output dataset is of size 1000. 
# 'dataset' combines x and c and then solve for each of 1000 problem instances 
# to add sols (decision variables) and objs (objective function value). 
dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)

# set data loader
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# build linear nn model for prediction

# init model
reg = LinearRegression(
    num_feat = NUM_FEAT,
    num_edge = NUM_EDGE
)

# init regret 
regret = pyepo.metric.regret(reg, optmodel, loader_test)

# train model

# init model SPO
reg = LinearRegression(
    num_feat = NUM_FEAT,
    num_edge = NUM_EDGE
)
spop = pyepo.func.SPOPlus(optmodel, processes=1)

# running the SPO model
loss_log_SPO, loss_log_regret_SPO = trainModel(
    reg=reg, loss_func=spop, method_name="spo+", opt_model=optmodel, 
    loader_train=loader_train, loader_test=loader_test, num_epochs=NUM_EPOCHS)

# output SPO results
df = pd.DataFrame(loss_log_SPO)
df.to_excel(OUTPUT_PATH + 'loss_logs_SPO.xlsx', index=False) 

df = pd.DataFrame(loss_log_regret_SPO)
df.to_excel(OUTPUT_PATH + 'loss_log_regret_SPO.xlsx', index=False) 
visLearningCurve(loss_log_SPO, loss_log_regret_SPO, OUTPUT_PATH + 'fig_SPO')

# init model 2s
# init MSE loss
mse = nn.MSELoss()
loss_log_2s, loss_log_regret_2s = trainModel(
    reg=reg, loss_func=mse, method_name="2s", opt_model=optmodel,
    loader_train=loader_train, loader_test=loader_test, num_epochs=NUM_EPOCHS)

# output 2s results
df = pd.DataFrame(loss_log_2s)
df.to_excel(OUTPUT_PATH + 'loss_logs_2s.xlsx', index=False) 

df = pd.DataFrame(loss_log_regret_2s)
df.to_excel(OUTPUT_PATH + 'loss_log_regret_2s.xlsx', index=False) 
visLearningCurve(loss_log_2s, loss_log_regret_2s, OUTPUT_PATH + 'fig_2s')
