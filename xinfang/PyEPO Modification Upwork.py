# -*- coding: utf-8 -*-
"""
Spyder Editor

This version adds the base case; which is to predcit then optimize. 
Also save x and c.


"""
# import lib
import pyepo
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

# set random seeds
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# number of training instances
num_data = 1000
# number of features x (to predict c)
num_feat = 5
# number of nodes in the network
num_node = 20 
# number of edges, converted to integer
num_edge = int(num_node * (num_node - 1) / 2)  

# neural network training epochs
# 10 for testing
num_epochs = 10

# A PyEPO function to generate synthetic data and features for travelling salesman
# See https://khalil-research.github.io/PyEPO/build/html/content/examples/data.html

# Note that x is of size (1000, 5) - 1000 samples/data points and 5 feature variables
# c is of size (1000, 190) - c is the target variable, which represents travel time on an edage
# For num_nodes = 20, the graph is typically a complete graph (meaning every node is connected 
# to every other node). E is given by the formula:n*(n - 1)/2 = 20*19/2 = 190
# So, the model uses 5 feature variables to predict 190 target variables. 

# Generate x, c. 
x, c = pyepo.data.tsp.genData(num_data, num_feat, num_node, deg=4, noise_width=0, seed=135)

# build optimization model by calling the single travelling salesman model

"""
Build a new class within grb.Single TSP.py based on mSkilledmTSP.py. 
So, I can build optmodel as follows. 
optmodel = pyepo.model.grb.MStspMTZModel(num_node) 
# optmodel.setObj(c[0]) # set objective function
# sol, obj = optmodel.solve() 
"""

optmodel = pyepo.model.grb.tspMTZModel(num_node)

# Below solves a single instance given c[0]
# optmodel.setObj(c[0]) # set objective function
# sol, obj = optmodel.solve() 

# split train test data
# x is feats (feature variables) and c is cost (target variable)
from sklearn.model_selection import train_test_split
x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=200, random_state=42)

# get optDataset
# build dataset: This class is Torch Dataset for optimization problems.
# output dataset is of size 1000. 
# 'dataset' combines x and c and then solve for each of 1000 problem instances 
# to add sols (decision variables) and objs (objective function value). 
dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)


# set data loader
batch_size = 30
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# build linear nn model for prediction
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_edge)

    def forward(self, x):
        out = self.linear(x)
        return out

# init model
reg = LinearRegression()

# init regret 
regret = pyepo.metric.regret(reg, optmodel, loader_test)

import time

# train model
# loader_train and loader_test are global variables; no need to pass as arguments
def trainModel(reg, loss_func, method_name, num_epochs=num_epochs, lr=1e-2):
    # set adam optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train mode
    reg.train()
    # init log
    loss_log = []
    #The initial regret is computed on the test set using pyepo.metric.regret().
    loss_log_regret = [pyepo.metric.regret(reg, optmodel, loader_test)]
    # init elpased time
    elapsed = 0
    for epoch in range(num_epochs):
        print("Start Epoch {:2}".format(epoch+1))
        # start timing
        tick = time.time()
        # load data
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            # cuda
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            
            cp = reg(x)
            if method_name == "spo+":
                loss = loss_func(cp, c, w, z)        
            if method_name == "ptb" or method_name == "pfy" or method_name == "nce" or method_name == "cmap":
                loss = loss_func(cp, w)
            if method_name == "dbb" or method_name == "nid":
                loss = loss_func(cp, c, z)
            #if method_name == "ltr": #learning to rank
            if method_name == "2s" or method_name == "ltr":
                loss = loss_func(cp, c)
            # backward pass
            optimizer.zero_grad()
            loss.mean().backward() #$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            optimizer.step()
            # record time
            tock = time.time()
            elapsed += tock - tick
            # log
            loss = loss.mean() #$$$$$$$$$$$$$$$$$$$$$$$$
            loss_log.append(loss.item())
        regret = pyepo.metric.regret(reg, optmodel, loader_test)
        loss_log_regret.append(regret)
        print("Epoch {:2},  Loss (Train): {:9.4f},  Regret (Validation): {:7.4f}%".format(epoch+1, loss.item(), regret*100))
    print("Total Elapsed Time: {:.2f} Sec.".format(elapsed))
    return loss_log, loss_log_regret 
    #loss_log is loss for training set and loss_log_regret is for testing set

#Define functions that visualize the learning curves
from matplotlib import pyplot as plt
from matplotlib import ticker 

def visLearningCurve(loss_log, loss_log_regret):
    # create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

    # draw plot for training loss
    ax1.plot(loss_log, color="c", lw=1)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.set_xlabel("Iters", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.set_title("Learning Curve on Training Set", fontsize=16)

    # draw plot for regret on test
    ax2.plot(loss_log_regret, color="royalblue", ls="--", alpha=0.7, lw=1)
    ax2.set_xticks(range(0, len(loss_log_regret), 2))
    ax2.tick_params(axis="both", which="major", labelsize=12)
    # Set y-axis limit dynamically based on the max value of loss_log_regret
    max_regret = max(loss_log_regret)+0.1
    ax2.set_ylim(0, max_regret)
   
    # Format y-axis labels as percentages
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))  # xmax=1 if data is normalized, adjust as needed
        
    ax2.set_xlabel("Epochs", fontsize=16)
    ax2.set_ylabel("Regret (%)", fontsize=16)
    ax2.set_title("Learning Curve on Test Set", fontsize=16)

    plt.show()

# init model SPO
import pandas as pd
reg = LinearRegression()
spop = pyepo.func.SPOPlus(optmodel, processes=1)

# running the SPO model
loss_log_SPO, loss_log_regret_SPO = trainModel(reg, loss_func=spop, method_name="spo+")

# output SPO results
df = pd.DataFrame(loss_log_SPO)
df.to_excel('loss_logs_SPO.xlsx', index=False) 

df = pd.DataFrame(loss_log_regret_SPO)
df.to_excel('loss_log_regret_SPO.xlsx', index=False) 
visLearningCurve(loss_log_SPO, loss_log_regret_SPO)

# init model 2s
# init MSE loss
mse = nn.MSELoss()
loss_log_2s, loss_log_regret_2s = trainModel(reg, loss_func=mse, method_name="2s")

# output 2s results
df = pd.DataFrame(loss_log_2s)
df.to_excel('loss_logs_2s.xlsx', index=False) 

df = pd.DataFrame(loss_log_regret_2s)
df.to_excel('loss_log_regret_2s.xlsx', index=False) 
visLearningCurve(loss_log_2s, loss_log_regret_2s)
