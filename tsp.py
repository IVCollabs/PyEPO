"""
Traveling salesman problem additional implementations for the PyEPO package.
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from pyepo.model.grb.tsp import tspABModel


class MStspMTZModel(tspABModel):
    """
    This class is optimization model for a multi skill traveling salesman problem based 
    on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """
    def __init__(self, num_nodes):
        """
        Args:
            num_nodes (int): number of nodes
        """
        self._loadInfo(num_nodes)
        super().__init__(num_nodes)

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
        x = m.addVars(self.num_nodes, self.num_nodes, self.num_salesmen, vtype=GRB.BINARY, name="x")
        u = m.addVars(self.num_nodes, name="u")
        # sense
        m.modelSense = GRB.MINIMIZE

        # Visiting constraints
        for i in range(1, self.num_nodes):
            # Each city (except depot) is visited exactly once
            m.addConstr(gp.quicksum(x[i, j, k] 
                                    for j in self.nodes if j != i 
                                    for k in range(self.num_salesmen)) == 1, name=f"VisitOnce_{i}")
            # Each city is departed from once
            m.addConstr(gp.quicksum(x[j, i, k] 
                                    for j in self.nodes if j != i 
                                    for k in range(self.num_salesmen)) == 1, name=f"DepartOnce_{i}")

        # Depot constraints
        for k in range(self.num_salesmen):
            m.addConstr(gp.quicksum(x[0, j, k] for j in range(1, self.num_nodes)) == 1, name=f"DepotExit_{k}")
            m.addConstr(gp.quicksum(x[j, 0, k] for j in range(1, self.num_nodes)) == 1, name=f"DepotEnter_{k}")

        # Flow conservation
        for k in range(self.num_salesmen):
            for i in self.nodes:
                A = gp.quicksum(x[i, j, k] for j in self.nodes if j != i)
                B = gp.quicksum(x[j, i, k] for j in self.nodes if j != i)
                m.addConstr(A == B, name=f"Flow_{i}_{k}")

        # Subtour elimination (MTZ constraints)
        for k in range(self.num_salesmen):
            for i in range(1, self.num_nodes):
                for j in range(1, self.num_nodes):
                    if i != j:
                        m.addConstr(
                            u[i] - u[j] + self.num_nodes * x[i, j, k] <= self.num_nodes - 1,
                            name=f"Subtour_{i}_{j}_{k}")

        for i in range(1, self.num_nodes):
            m.addConstr(u[i] >= 1, name=f"uLower_{i}")
            m.addConstr(u[i] <= self.num_nodes - 1, name=f"uUpper_{i}")

        # Skill constraints
        for k in range(self.num_salesmen):
            for i in range(1, self.num_nodes):
                if i not in self.skills[k]:
                    for j in self.nodes:
                        m.addConstr(x[i, j, k] == 0, name=f"SkillOut_{i}_{j}_{k}")
                        m.addConstr(x[j, i, k] == 0, name=f"SkillIn_{j}_{i}_{k}")

        # Maximum cities per salesman
        for k in range(self.num_salesmen):
            m.addConstr(gp.quicksum(x[i, j, k] 
                                    for i in self.nodes 
                                    for j in range(1, self.num_nodes) 
                                    if i != j) <= self.max_city, name=f"MaxTour_{k}")
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

        def criar_matriz_simetrica(vetor):
            # Determinar o tamanho da matriz k
            N = len(vetor)
            k = int((1 + np.sqrt(1 + 8 * N)) / 2)  # Resolve N = k*(k-1)/2 para k
            
            # Inicializa uma matriz kxk com zeros
            matriz = np.zeros((k, k))
            
            # Preenche o triângulo superior (excluindo a diagonal principal)
            idx = 0  # Índice para percorrer o vetor
            for i in range(k):
                for j in range(i + 1, k):
                    matriz[i, j] = vetor[idx]
                    idx += 1
            
            # Reflete os valores no triângulo inferior
            for i in range(k):
                for j in range(i + 1, k):
                    matriz[j, i] = matriz[i, j]
            
            return matriz

        # Exemplo de uso
        # vetor = [1, 2, 3, 4, 5, 6]  # Vetor de 6 elementos (k=4)
        c_matrix = criar_matriz_simetrica(c)
        # print(matriz_simetrica)

        aux = []
        for k in range(self.num_salesmen):
            
            for i in self.nodes:
                for j in self.nodes:
                    if i != j:
                        aux.append(self.x[i, j, k] *c_matrix[i][j])


        obj = gp.quicksum(aux)
        # obj = gp.quicksum(self.distances[i][j] * self.x[i, j, k] 
        #                   for i in self.nodes 
        #                   for j in self.nodes 
        #                   for k in range(self.num_salesmen))
        
        self._model.setObjective(obj)

    def solve(self):

        sol, _ = super().solve()

        sol =  [sum(sol[i:i + self.num_salesmen]) for i in range(0, len(sol), self.num_salesmen)]

        sol_matrix = [sol[i:i + self.num_nodes] for i in range(0, len(sol), self.num_nodes)]

        def soma_simetricos(matriz):
            n = len(matriz)
            vetor_resultante = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    soma = matriz[i][j] + matriz[j][i]
                    vetor_resultante.append(soma)
            
            return vetor_resultante

        sol = soma_simetricos(sol_matrix)

        return sol, _

    def _loadInfo(self, num_nodes):
        """
        A method to load additional information from external files for the model
        
        Args:
            num_nodes (int): number of nodes
        """
        # TODO: Read from an additional file
        # Number of salesmen
        self.num_salesmen = 2  
        # Maximum cities per salesman -  equitable work load???
        self.max_city = 3

        # Skill constraints: which salesmen can visit which cities (excluding depot)
        self.skills = {
            0: [1, 2, 3],  # Salesman 0 can visit cities 1, 2, 3
            1: [3, 4],     # Salesman 1 can visit cities 3, 4
        }

        self.distances = np.random.randint(1, 20, size=(num_nodes, num_nodes))
        np.fill_diagonal(self.distances, 0)
        self.distances = (self.distances + self.distances.T) // 2  # Make it symmetric
        
    def relax(self):
        """
        A method to get linear relaxation model
        """
        raise RuntimeError("Relaxation Model not implemented.")
