#https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/03%20Training%20and%20Testing.ipynb

#https://khalil-research.github.io/PyEPO/build/html/content/examples/training.html#training-with-spo

import os
import numpy as np
import pandas as pd
import pyepo
import random
import torch

from torch import nn
from torch.utils.data import DataLoader

from classes_and_methods import LinearRegression, trainModel, plot_mse_and_regret_side_by_side
from tsp import MStspMTZModel

# Number of locations 
# (1st location is depot (centriod of patient locations); rest for patients)
num = 11 

# Parameters
#NUM_DATA = 10   # number of training instances
NUM_FEAT = 8     # number of features x (to predict c)    
NUM_EPOCHS = 30 # number of epochs for training
BATCH_SIZE = 16  # batch size for training
OUTPUT_PATH = "outputs/"                     # output folder
INPUT_PATH = "input_data"                    # input folder
SKILL_MAPPING_PATH = "/skill_mapping11.xlsx"   # skill file (index of salesman is TSP)
COORDINATES_PATH = "coordinates11.csv"         # coodinates file (columns: Latitude and Longitude)

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


# x (feature variables) are the same for different number of locations)
x_train = pd.read_csv(os.path.join(INPUT_PATH, "x_train.csv"), index_col=0).to_numpy(dtype=np.float64)
x_test = pd.read_csv(os.path.join(INPUT_PATH, "x_test.csv"), index_col=0).to_numpy(dtype=np.float64)
c_train = pd.read_csv(os.path.join(INPUT_PATH, f"c_train_{num}.csv"), index_col=0).to_numpy(dtype=np.float64)
c_test = pd.read_csv(os.path.join(INPUT_PATH, f"c_test_{num}.csv"), index_col=0).to_numpy(dtype=np.float64)

# Get optDataset
# Build dataset: This class is Torch Dataset for optimization problems.
# 'dataset' combines x and c and then solve for each of 1000 problem instances 
# to add sols (decision variables) and objs (objective function value). 

dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)

# Set data loader
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# Collect min regrets for all seeds
min_regrets_list = []
      
# Loop over seeds
for SEED in range(1, 11):
    print(f"--- Running for SEED = {SEED} ---")

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
         
    # 2S training
    torch.manual_seed(SEED)
    reg = LinearRegression(num_feat=NUM_FEAT, num_edge=NUM_EDGE)
    mse = nn.MSELoss()
   
    loss_log_regret_2s, mse_log_2s= trainModel(reg=reg, loss_func=mse, method_name="2s",
                                                 opt_model=optmodel, loader_train=loader_train,
                                               loader_test=loader_test, num_epochs=NUM_EPOCHS)
       
    # SPO+ training
    spop = pyepo.func.SPOPlus(optmodel, processes=1)
    torch.manual_seed(SEED)
    reg = LinearRegression(num_feat=NUM_FEAT, num_edge=NUM_EDGE)
    
    loss_log_regret_SPO, mse_log_SPO = trainModel(reg=reg, loss_func=spop, method_name="spo+",
                                                   opt_model=optmodel, loader_train=loader_train,
                                               loader_test=loader_test, num_epochs=NUM_EPOCHS)
     
    # Convert logs to DataFrames
    df_2s_mse = pd.DataFrame(mse_log_2s, columns=["mse_2s"])
    df_2s_regret = pd.DataFrame(loss_log_regret_2s, columns=["decision_regret_2s", "timestamp_2s"])
    
    df_spo_mse = pd.DataFrame(mse_log_SPO, columns=["mse_spo"])
    df_spo_regret = pd.DataFrame(loss_log_regret_SPO, columns=["decision_regret_spo", "timestamp_spo"])
    
    # Combine all columns side-by-side
    df_all = pd.concat([df_2s_mse, df_2s_regret, df_spo_mse, df_spo_regret], axis=1)
    
    # Write to a single sheet
    df_all.to_excel(
        f"{OUTPUT_PATH}all_logs_seed{SEED}_location{num}.xlsx",
        index=True
    )

    min_spo_tuple = min(loss_log_regret_SPO, key=lambda x: x[0])
    min_2s_tuple  = min(loss_log_regret_2s, key=lambda x: x[0])
    
    min_regrets_list.append({
        'seed': SEED,
        'min_regret_SPO': min_spo_tuple[0],
        'time_min_SPO':   min_spo_tuple[1],
        'min_regret_2s':  min_2s_tuple[0],
        'time_min_2s':    min_2s_tuple[1],
    })

    plot_mse_and_regret_side_by_side(df_all, SEED, num, OUTPUT_PATH)
