
import numpy as np

from tsp import MStspMTZModel
from xinfang.plot_routes import plot_salesman_routes

# Seed for replicability
np.random.seed(42) 

# Initializes the model
optmodel = MStspMTZModel(num_nodes=5)

# Sets the objective function
cost = None
optmodel.setObj(cost)

# Solves the model
sol, obj = optmodel.solve()
print('Obj: {}'.format(obj))

# Extract routes from solution
selected_nodes = [
    list(optmodel.x.keys())[i] 
    for i, val in enumerate(sol) 
    if val>0.99 # Include only active nodes
] 

routes = {}
for (i, j, k) in selected_nodes:
    if k not in routes:
        routes[k] = []
    routes[k].append((i, j))
print(routes)

coordinates = {
        0: (0, 0),  # Depot
        1: (2, 1),
        2: (1, 3),
        3: (3, 2),
        4: (4, 4),
    }

# Transform the distance matrix into a dictionary
dist = {}
n = len(optmodel.distances)
for i in range(n):
    for j in range(n):
        if i != j:  
            dist[(i, j)] = optmodel.distances[i][j]
            
plot_salesman_routes(routes, coordinates, optmodel.num_salesmen, dist) 


# based on: https://github.com/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb