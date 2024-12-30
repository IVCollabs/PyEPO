
from tsp import MStspMTZModel, distances, salesmen
from xinfang.plot_routes import plot_salesman_routes


optmodel = MStspMTZModel(num_nodes=5)
sol, obj = optmodel.solve()
print('Obj: {}'.format(obj))

# Extract routes from solution
routes = {}
for i, (i, j, k) in enumerate(optmodel.x.keys()):
    if sol[i] > 0.99: # Include only active routes
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
n = len(distances)
for i in range(n):
    for j in range(n):
        if i != j:  
            dist[(i, j)] = distances[i][j]
            
plot_salesman_routes(routes, coordinates, salesmen, dist) 


# based on: https://github.com/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb