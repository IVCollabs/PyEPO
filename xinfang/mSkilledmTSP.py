# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 05:53:06 2024

@author: xfwang
"""

# import lib
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Parameters
# Number of nodes (including depot)
n = 5  
# Number of salesmen
s = 2  
# Maximum cities per salesman -  equitable work load???
q = 3

# Generate random distance matrix (symmetric with zeros on the diagonal)
np.random.seed(42)
distances = np.random.randint(1, 20, size=(n, n))
np.fill_diagonal(distances, 0)
distances = (distances + distances.T) // 2  # Make it symmetric

# Skill constraints: which salesmen can visit which cities (excluding depot)
skills = {
    0: [1, 2, 3],  # Salesman 0 can visit cities 1, 2, 3
    1: [3, 4],     # Salesman 1 can visit cities 3, 4
}



# Init Model
model = gp.Model("mTSP_with_Skills")

# mTSP formulation 
# Variables
x = model.addVars(n, n, s, vtype=GRB.BINARY, name="x")
u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")

# Objective: Minimize travel cost
model.setObjective(gp.quicksum(distances[i][j] * x[i, j, k] for i in range(n) for j in range(n) for k in range(s)), GRB.MINIMIZE)

# Constraints

# Each city (except depot) is visited exactly once
for i in range(1, n):
    model.addConstr(gp.quicksum(x[i, j, k] for j in range(n) if j != i for k in range(s)) == 1, name=f"VisitOnce_{i}")

# Each city is departed from once
for i in range(1, n):
    model.addConstr(gp.quicksum(x[j, i, k] for j in range(n) if j != i for k in range(s)) == 1, name=f"DepartOnce_{i}")

# Depot constraints
for k in range(s):
    model.addConstr(gp.quicksum(x[0, j, k] for j in range(1, n)) == 1, name=f"DepotExit_{k}")
    model.addConstr(gp.quicksum(x[j, 0, k] for j in range(1, n)) == 1, name=f"DepotEnter_{k}")

# Flow conservation
for k in range(s):
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j, k] for j in range(n) if j != i) == gp.quicksum(x[j, i, k] for j in range(n) if j != i), name=f"Flow_{i}_{k}")

# Subtour elimination (MTZ constraints)
for k in range(s):
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j, k] <= n - 1, name=f"Subtour_{i}_{j}_{k}")

for i in range(1, n):
    model.addConstr(u[i] >= 1, name=f"uLower_{i}")
    model.addConstr(u[i] <= n - 1, name=f"uUpper_{i}")

# Skill constraints
for k in range(s):
    for i in range(1, n):
        if i not in skills[k]:
            for j in range(n):
                model.addConstr(x[i, j, k] == 0, name=f"SkillOut_{i}_{j}_{k}")
                model.addConstr(x[j, i, k] == 0, name=f"SkillIn_{j}_{i}_{k}")

# Maximum cities per salesman
for k in range(s):
    model.addConstr(gp.quicksum(x[i, j, k] for i in range(n) for j in range(1, n) if i != j) <= q, name=f"MaxTour_{k}")

# Solve
model.optimize()

all_vars = model.getVars()
values = model.getAttr("X", all_vars)
names = model.getAttr("VarName", all_vars)

print("\nNon-zero decision variable values:")
for name, val in zip(names, values):
    if abs(val) > 1e-6:  # Filter to show only significant non-zero values
        print(f"{name} = {val}")

from plot_routes import plot_salesman_routes

routes = {}

# Extract routes from x
for (i, j, k), value in x.items():
    if value.X > 0.5:  # Include only active routes
        if k not in routes:
            routes[k] = []
        routes[k].append((i, j))

coordinates = {
        0: (0, 0),  # Depot
        1: (2, 1),
        2: (1, 3),
        3: (3, 2),
        4: (4, 4),
    }

# Transform the distance matrix into a dictionary
dist = {}
n = len(distances)
for i in range(n):
    for j in range(n):
        if i != j:  
            dist[(i, j)] = distances[i][j]
            
plot_salesman_routes(routes, coordinates, s, dist) 
     
 
    

