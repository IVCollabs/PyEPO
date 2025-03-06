"""
Traveling salesman problem additional implementations for the PyEPO package.
"""

import os
import pandas as pd
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from pyepo.model.grb.tsp import tspABModel
from classes_and_methods import haversine


class MStspMTZModel(tspABModel):
    """
    This class is optimization model for a multi skill traveling salesman problem based 
    on Miller-Tucker-Zemlin (MTZ) formulation.

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_nodes (int): Number of nodes
        edges (list): List of edge index
    """
    def __init__(self,input_path,skill_path,coodinates_path):
        self._loadInfo(input_path,skill_path,coodinates_path)
        super().__init__(len(self.coordinates))

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
        """
        A method to set the objective function for the optimization model.

        This method constructs a symmetric cost matrix from the given cost vector `c`
        and uses it to define the objective function. The objective function represents
        the total cost of all paths traversed by the salesmen, excluding self-loops
        (i.e., paths from a node to itself).

        Args:
            c (list): A cost vector representing the costs of paths between nodes.
                    The length of `c` should satisfy N = k*(k-1)/2, where k is the
                    number of nodes.
        """

        # Determine the size of the cost matrix `k` based on the length of the cost vector `c`
        # The formula solves N = k*(k-1)/2 for k, where N is the length of `c`
        N = len(c)
        k = int((1 + np.sqrt(1 + 8 * N)) / 2)

        # Initialize a kxk matrix with zeros to store the costs
        c_matrix = np.zeros((k, k))

        # Fill the upper triangle of the matrix (excluding the main diagonal)
        # using the values from the cost vector `c`
        idx = 0  # Index to traverse the cost vector
        for i in range(k):
            for j in range(i + 1, k):
                c_matrix[i, j] = c[idx]
                idx += 1

        # Mirror the values from the upper triangle to the lower triangle
        # to make the matrix symmetric
        for i in range(k):
            for j in range(i + 1, k):
                c_matrix[j, i] = c_matrix[i, j]

        # Calculate the total cost by summing the costs of all paths traversed by the salesmen
        # Exclude self-loops (i.e., paths where i == j)
        aux = []
        for k in range(self.num_salesmen):
            for i in self.nodes:
                for j in self.nodes:
                    if i != j:
                        aux.append(self.x[i, j, k] * c_matrix[i][j])

        # Define the objective function as the sum of all costs
        obj = gp.quicksum(aux)

        # Set the objective function for the model
        self._model.setObjective(obj)
    
    def solve(self):
        """
        Solves the problem and processes the solution to aggregate the results.

        This method overrides the parent class's solve method to post-process the solution.
        It aggregates the solution by summing the contributions of each salesman and
        removes self-loops (i.e., paths from a node to itself). The final solution
        represents the total number of times each path was traversed by all salesmen.

        Returns:
            tuple: A tuple containing the processed solution
                returned by the parent class's solve method.
        """

        # Call the parent class's solve method to get the initial solution
        sol, _ = super().solve()

        # Aggregate the solution by summing the contributions of each salesman
        # This is done because we are interested in the total number of times each path
        # was traversed, regardless of which salesman traversed it.
        sol = [sum(sol[i:i + self.num_salesmen]) for i in range(0, len(sol), self.num_salesmen)]

        # Reshape the solution into a matrix where each row represents a node
        # and each column represents the connections from that node.
        sol_matrix = [sol[i:i + self.num_nodes] for i in range(0, len(sol), self.num_nodes)]

        # Initialize a list to store the updated solution
        updated_sol = []

        # Iterate over the solution matrix to compute the sum of paths between nodes
        # while ignoring self-loops (i.e., paths from a node to itself).
        for i in range(len(sol_matrix)):
            for j in range(i + 1, len(sol_matrix)):
                # Sum the paths from node i to node j and from node j to node i
                soma = sol_matrix[i][j] + sol_matrix[j][i]
                updated_sol.append(soma)

        # Update the solution with the processed results
        sol = updated_sol

        # Return the processed solution and any additional metadata
        return sol, _

    def _loadInfo(
            self,
            input_path: str,
            skill_path: str,
            coodinates_path: str):
        """
        A method to load additional information from external files for the model.

        This method reads data from three specified file paths: one for general input data,
        one for skill-related data, and one for coordinate data. These files are used to
        populate the model with the necessary information for further processing.

        Args:
            input_path (str): The file path to the input data folder.

            skill_path (str): The file path to the skill data file. This file contains information
                            about the skills required or available at each node, which is used
                            to match nodes with the appropriate salesmen or resources.

            coordinates_path (str): The file path to the coordinates data file. This file contains
                                    the spatial coordinates (e.g., latitude and longitude) of each
                                    node, which are used for distance calculations or visualization.

        """

        # Loading skills
        data = pd.read_excel(input_path+skill_path)
        data = data.set_index('TSP')
        self.skills = {}
        for i in range(len(data)):
            self.skills[i] = [idx for idx, j in enumerate(data.iloc[i]) if j == 1]
        self.num_salesmen = len(self.skills)
        self.max_city = len(data.iloc[0])

        # Loading coordinates
        data = pd.read_csv(os.path.join(input_path, coodinates_path))
        coordinates = [(lat, lon) for lat, lon in zip(data['Latitude'], data['Longitude'])]

        # Filtering the same coordinates 
        coordinates = list(set(coordinates))

        # TODO: The type of the data is not the same was before
        # Calculating the distances
        DISTANCE_PRECISION = 1
        distance_matrix = []
        for cord1 in coordinates:
            cord_list = []
            for cord2 in coordinates:
                distance = haversine(cord1, cord2)

                # Multiply the distance and round it to the nearest integer
                # distance = round(distance * DISTANCE_PRECISION)
                distance = distance * DISTANCE_PRECISION

                cord_list.append(distance)
            distance_matrix.append(cord_list)

        # Creating the atributes
        self.coordinates = coordinates
        self.distances = distance_matrix
        
    def relax(self):
        """
        A method to get linear relaxation model
        """
        raise RuntimeError("Relaxation Model not implemented.")
