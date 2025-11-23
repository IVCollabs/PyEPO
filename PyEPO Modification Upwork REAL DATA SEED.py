"""Main script to run custom model.

Important links: 
https://colab.research.google.com/github/khalil-research/PyEPO/blob/main/notebooks/03%20Training%20and%20Testing.ipynb
https://khalil-research.github.io/PyEPO/build/html/content/examples/training.html
"""

import os
import numpy as np
import pandas as pd
import pyepo
import random
import torch

from torch.utils.data import DataLoader

from classes_and_methods import LinearRegression, trainModel, plot_mse_and_regret_side_by_side
from loss_functions import LossMethods
from tsp import MStspMTZModel


# Number of locations 
# (1st location is depot (centriod of patient locations); rest for patients)
num = 11 

# Parameters
#NUM_DATA = 10   # number of training instances
NUM_FEAT = 8     # number of features x (to predict c)    
NUM_EPOCHS = 2 # number of epochs for training
BATCH_SIZE = 10  # batch size for training
OUTPUT_PATH = "outputs/"                     # output folder
INPUT_PATH = "input_data"                    # input folder
SKILL_MAPPING_PATH = f"/skill_mapping{num}.xlsx"   # skill file (index of salesman is TSP)
COORDINATES_PATH = f"coordinates{num}.csv"         # coodinates file (columns: Latitude and Longitude)

METHODS = ['2s', 'spo+'] #loss functionS to be used 

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
for SEED in range(1, 3):
    print(f"--- Running for SEED = {SEED} ---")

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Init linear nn model for prediction
    reg = LinearRegression(
        num_feat = NUM_FEAT,
        num_edge = NUM_EDGE
    )

    ### Training models
    method_dict = LossMethods(optmodel, dataset_train).get_loss_dict()
    df_results=[]

    for method in METHODS:
        # Running the model
        print(method)
        loss_log_regret, mse_log = trainModel(
            reg=reg, loss_func=method_dict[method], method_name=method, opt_model=optmodel, 
            loader_train=loader_train, loader_test=loader_test, num_epochs=NUM_EPOCHS)

        ### Saving the results
        df_mse = pd.DataFrame(mse_log, columns=[f"mse_{method}"])
        df_regret = pd.DataFrame(loss_log_regret, columns=[f"decision_regret_{method}", f"timestamp_{method}"])
        df_results.append(df_mse)
        df_results.append(df_regret)
        
    # Combine all columns side-by-side
    df_all = pd.concat(df_results, axis=1)
    
    # Write to a single sheet
    df_all.to_excel(
        f"{OUTPUT_PATH}all_logs_seed{SEED}_location{num}.xlsx",
        index=True
    )

    plot_mse_and_regret_side_by_side(df_all, SEED, num, METHODS, OUTPUT_PATH)
