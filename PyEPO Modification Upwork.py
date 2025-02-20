import os
import numpy as np
import pandas as pd
import pyepo
import random
import torch

from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from classes_and_methods import LinearRegression, trainModel, visLearningCurve
from tsp import MStspMTZModel

# Parameters
NUM_SALESMAN = 2 # number of salesman
NUM_DATA = 100   # number of training instances
NUM_FEAT = 5     # number of features x (to predict c)    
NUM_EPOCHS = 100 # number of epochs for training
BATCH_SIZE = 10  # batch size for training
OUTPUT_PATH = "outputs/"                     # output folder
INPUT_PATH = "input_data"                    # input folder
SKILL_MAPPING_PATH = "/skill_mapping.xlsx"   # skill file (index of salesman is TSP)
COORDINATES_PATH = "coordinates.csv"         # coodinates file (columns: Latitude and Longitude)

# Verifies if outputs path folder exist,
# if not, creates one.
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Build optimization model by calling the single travelling salesman model
optmodel = MStspMTZModel(INPUT_PATH,SKILL_MAPPING_PATH,COORDINATES_PATH)

NUM_NODE = len(optmodel.coordinates) # number of nodes in the network

# Number of edges, converted to integer
NUM_EDGE = int(NUM_NODE * (NUM_NODE - 1) / 2)  

# A PyEPO function to generate synthetic data and features for travelling salesman
# See https://khalil-research.github.io/PyEPO/build/html/content/examples/data.html
# Generate x, c. 
x, c = pyepo.data.tsp.genData(NUM_DATA, NUM_FEAT, NUM_NODE, deg=4, noise_width=0, seed=135)

# Preparing the distance vector to update c
dist_matrix = np.triu(optmodel.distances)
dist_vector = dist_matrix[dist_matrix!=0]
c = dist_vector/c

### Preparing the data

# Split train test data
# x is feats (feature variables) and c is cost (target variable)
x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=0.2, random_state=42)

# Get optDataset
# Build dataset: This class is Torch Dataset for optimization problems.
# 'dataset' combines x and c and then solve for each of 1000 problem instances 
# to add sols (decision variables) and objs (objective function value). 
dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)

# Set data loader
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

### Building model

# Init linear nn model for prediction
reg = LinearRegression(
    num_feat = NUM_FEAT,
    num_edge = NUM_EDGE
)

# Init regret 
regret = pyepo.metric.regret(reg, optmodel, loader_test)
print("Regret benchmark: ", regret)

### Training models

# Init model SPO
reg = LinearRegression(
    num_feat = NUM_FEAT,
    num_edge = NUM_EDGE
)
spop = pyepo.func.SPOPlus(optmodel, processes=1)

# Running the SPO model
loss_log_SPO, loss_log_regret_SPO = trainModel(
    reg=reg, loss_func=spop, method_name="spo+", opt_model=optmodel, 
    loader_train=loader_train, loader_test=loader_test, num_epochs=NUM_EPOCHS)

# Init model 2s and MSE loss
mse = nn.MSELoss()
loss_log_2s, loss_log_regret_2s = trainModel(
    reg=reg, loss_func=mse, method_name="2s", opt_model=optmodel,
    loader_train=loader_train, loader_test=loader_test, num_epochs=NUM_EPOCHS)

### Saving the results

# Output SPO results
df = pd.DataFrame(loss_log_SPO)
df.to_excel(OUTPUT_PATH + 'loss_logs_SPO.xlsx', index=False) 

df = pd.DataFrame(loss_log_regret_SPO)
df.to_excel(OUTPUT_PATH + 'loss_log_regret_SPO.xlsx', index=False) 
visLearningCurve(loss_log_SPO, loss_log_regret_SPO, OUTPUT_PATH + 'fig_SPO')

# Output 2s results
df = pd.DataFrame(loss_log_2s)
df.to_excel(OUTPUT_PATH + 'loss_logs_2s.xlsx', index=False) 

df = pd.DataFrame(loss_log_regret_2s)
df.to_excel(OUTPUT_PATH + 'loss_log_regret_2s.xlsx', index=False) 
visLearningCurve(loss_log_2s, loss_log_regret_2s, OUTPUT_PATH + 'fig_2s')
