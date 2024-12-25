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
        x = m.addVars(self.nodes, self.nodes, salesmen, vtype=GRB.BINARY, name="x")
        u = m.addVars(self.nodes, name="u")
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstrs(x.sum("*", j) == 1 for j in self.nodes)
        m.addConstrs(x.sum(i, "*") == 1 for i in self.nodes)
        m.addConstrs(u[j] - u[i] >=
                     len(self.nodes) * (x[i,j] - 1) + 1
                     for (i,j) in directed_edges
                     if (i != 0) and (j != 0))
        return m, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (list): cost vector
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        obj = gp.quicksum(c[k] * (self.x[i,j] + self.x[j,i])
                          for k, (i,j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self):
        """
        A method to solve model
        """
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i,j) in enumerate(self.edges):
            if self.x[i,j].x > 1e-2 or self.x[j,i].x > 1e-2:
                sol[k] = 1
        return sol, self._model.objVal

    def addConstr(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            optModel: new model with the added constraint
        """
        if len(coefs) != self.num_cost:
            raise ValueError("Size of coef vector cannot cost.")
        # copy
        new_model = self.copy()
        # add constraint
        new_model._model.addConstr(
            gp.quicksum(coefs[k] * (new_model.x[i,j] + new_model.x[j,i])
                        for k, (i,j) in enumerate(new_model.edges)) <= rhs)
        return new_model

    def relax(self):
        """
        A method to get linear relaxation model
        """
        raise RuntimeError("Relaxation Model not implemented.")
