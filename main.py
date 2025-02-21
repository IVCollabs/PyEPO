"""Main script to run custom model.

Based on a PyEPO example in: 
https://github.com/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb
"""

import numpy as np

from plot import salemen_routes_plot
from tsp import MStspMTZModel

# Seed for replicability
np.random.seed(42) 

# Initializes the model
optmodel = MStspMTZModel()

# Sets the objective function
dist_matrix = np.triu(optmodel.distances)
cost = dist_matrix[dist_matrix!=0]
optmodel.setObj(cost)

# Solves the model
sol, obj = optmodel.solve()
print('Obj: {}'.format(obj))

# Extract routes from solution
selected_nodes = [
    idx
    for idx, val in optmodel.x.items() 
    if val.X>0.99 # Include only active nodes
] 

routes = {}
for (i, j, k) in selected_nodes:
    if k not in routes:
        routes[k] = []
    routes[k].append((i, j))
print(routes)

coordinate_dict = {}
for idx, coord in enumerate(optmodel.coordinates):
    coordinate_dict[idx] = coord

# Transform the distance matrix into a dictionary
dist = {}
n = len(optmodel.distances)
for i in range(n):
    for j in range(n):
        if i != j:  
            dist[(i, j)] = optmodel.distances[i][j]
            
salemen_routes_plot(coordinate_dict, routes)
