"""
Traveling salesman problem additional implementations for the PyEPO package.
"""


import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pyepo.model.grb.tsp import tspABModel

# TODO: Read from an additional file
# Number of salesmen
salesmen = 2  
# Maximum cities per salesman -  equitable work load???
q = 3

nodes = 5

# Skill constraints: which salesmen can visit which cities (excluding depot)
skills = {
    0: [1, 2, 3],  # Salesman 0 can visit cities 1, 2, 3
    1: [3, 4],     # Salesman 1 can visit cities 3, 4
}
# Generate random distance matrix (symmetric with zeros on the diagonal)
np.random.seed(42) #TODO: Remover seed depois

#TODO: nodes é o mesmo de self.nodes, talvez seja melhor a leitura desses arquivos ser feita dentro de um método novo da classe
distances = np.random.randint(1, 20, size=(nodes, nodes))
np.fill_diagonal(distances, 0)
distances = (distances + distances.T) // 2  # Make it symmetric


class MStspMTZModel(tspABModel):
    """
    This class is optimization model for a multi skill traveling salesman problem based on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """
    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("multi_skills_tsp")
        # turn off output
        m.Params.outputFlag = 0
        # variables
        x = m.addVars(self.num_nodes, self.num_nodes, salesmen, vtype=GRB.BINARY, name="x")
        u = m.addVars(self.num_nodes, name="u")
        # sense
        m.modelSense = GRB.MINIMIZE

        # constraints
        # Each city (except depot) is visited exactly once
        for i in range(1, self.num_nodes):
            m.addConstr(gp.quicksum(x[i, j, k] for j in self.nodes if j != i for k in range(salesmen)) == 1, name=f"VisitOnce_{i}")

        # Each city is departed from once
        for i in range(1, self.num_nodes):
            m.addConstr(gp.quicksum(x[j, i, k] for j in self.nodes if j != i for k in range(salesmen)) == 1, name=f"DepartOnce_{i}")

        # Depot constraints
        for k in range(salesmen):
            m.addConstr(gp.quicksum(x[0, j, k] for j in range(1, self.num_nodes)) == 1, name=f"DepotExit_{k}")
            m.addConstr(gp.quicksum(x[j, 0, k] for j in range(1, self.num_nodes)) == 1, name=f"DepotEnter_{k}")

        # Flow conservation
        for k in range(salesmen):
            for i in self.nodes:
                m.addConstr(gp.quicksum(x[i, j, k] for j in self.nodes if j != i) == gp.quicksum(x[j, i, k] for j in self.nodes if j != i), name=f"Flow_{i}_{k}")

        # Subtour elimination (MTZ constraints)
        for k in range(salesmen):
            for i in range(1, self.num_nodes):
                for j in range(1, self.num_nodes):
                    if i != j:
                        m.addConstr(u[i] - u[j] + self.num_nodes * x[i, j, k] <= self.num_nodes - 1, name=f"Subtour_{i}_{j}_{k}")

        for i in range(1, self.num_nodes):
            m.addConstr(u[i] >= 1, name=f"uLower_{i}")
            m.addConstr(u[i] <= self.num_nodes - 1, name=f"uUpper_{i}")

        # Skill constraints
        for k in range(salesmen):
            for i in range(1, self.num_nodes):
                if i not in skills[k]:
                    for j in self.nodes:
                        m.addConstr(x[i, j, k] == 0, name=f"SkillOut_{i}_{j}_{k}")
                        m.addConstr(x[j, i, k] == 0, name=f"SkillIn_{j}_{i}_{k}")

        # Maximum cities per salesman
        for k in range(salesmen):
            m.addConstr(gp.quicksum(x[i, j, k] for i in self.nodes for j in range(1, self.num_nodes) if i != j) <= q, name=f"MaxTour_{k}")
        return m, x

    def setObj(self, c):
        #TODO: This needs to be updated to minimize traval cost
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        # if len(c) != self.num_cost:
        #     raise ValueError("Size of cost vector cannot match vars.")
        # obj = gp.quicksum(c[k] * (self.x[i,j] + self.x[j,i])
        #                   for k, (i,j) in enumerate(self.edges))
        obj = gp.quicksum(distances[i][j] * self.x[i, j, k] for i in self.nodes for j in self.nodes for k in range(salesmen))
        
        self._model.setObjective(obj)

    # def solve(self):
    #     """
    #     A method to solve model
    #     """
    #     #TODO: This needs to be updated since the format of the decision variable has changed
    #     self._model.update()
    #     self._model.optimize()
    #     sol = np.zeros(self.num_cost, dtype=np.uint8)
    #     for k, (i,j) in enumerate(self.edges):
    #         if self.x[i,j].x > 1e-2 or self.x[j,i].x > 1e-2:
    #             sol[k] = 1
    #     return sol, self._model.objVal
    
    def relax(self):
        """
        A method to get linear relaxation model
        """
        raise RuntimeError("Relaxation Model not implemented.")
