# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:15:44 2024

@author: xfwang
"""

def plot_salesman_routes(routes, coordinates, num_salesmen, dist):
    """
    Plots the routes of salesmen on a graph with nodes color-coded by the salesman visiting them
    and optional distance labels on edges.

    Parameters:
    - routes (dict): Dictionary where keys are salesmen (int) and values are lists of edges [(i, j), ...].
    - coordinates (dict): Dictionary where keys are node indices and values are (x, y) coordinates.
    - num_salesmen (int): Total number of salesmen for color generation.
    - dist (dict): Distance matrix as a dictionary {(i, j): distance, ...}.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Generate colors for salesmen
    salesman_colors = {k: plt.cm.get_cmap("tab10")(k) for k in range(num_salesmen)}

    # Map nodes to salesmen based on routes
    node_salesman = {node: [] for node in coordinates.keys()}
    for salesman, edges in routes.items():
        for i, j in edges:
            if j != 0:  # Skip depot
                node_salesman[j].append(salesman)

    # Determine node colors based on the salesman who visited
    node_colors = []
    for node in coordinates.keys():
        if node == 0:  # Depot node with no fill color
            node_colors.append("white")
        else:
            # Use the first salesman who visited the node for color
            node_colors.append(salesman_colors[node_salesman[node][0]])

    # Plot the routes
    plt.figure(figsize=(8, 8))
    G = nx.DiGraph()

    # Add nodes and positions
    for node, coord in coordinates.items():
        G.add_node(node, pos=coord)

    pos = nx.get_node_attributes(G, 'pos')

    # Add edges and plot for each salesman
    for salesman, edges in routes.items():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=[salesman_colors[salesman]],
            width=2.5,
            arrowsize=15,
            label=f"Salesman {salesman + 1}",  # Display "Salesman 1", "Salesman 2", etc.
        )

        # Annotate distances on edges
        edge_labels = {(i, j): f"{dist[i, j]:.2f}" for i, j in edges if (i, j) in dist}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Plot nodes with updated colors and black borders
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    # Add legend
    handles = [plt.Line2D([0], [0], color=salesman_colors[salesman], lw=2, label=f"Salesman {salesman + 1}")
               for salesman in routes]
    plt.legend(handles=handles, loc="upper left")

    plt.title("Routes by Salesman with Distances")
    plt.axis("off")
    plt.show()

"""
# Example of usage
if __name__ == "__main__":
    # Example data
    coordinates = {
        0: (0, 0),  # Depot
        1: (2, 1),
        2: (1, 3),
        3: (3, 2),
        4: (4, 4),
    }

    routes = {
        0: [(0, 2), (2, 1), (1, 3), (3, 0)],
        1: [(0, 4), (4, 0)],
    }

    num_salesmen = 2

    # Example distance dictionary
    dist = {
        (0, 2): 3.16,
        (2, 1): 2.24,
        (1, 3): 2.24,
        (3, 0): 3.61,
        (0, 4): 5.66,
        (4, 0): 5.66,
    }

    # Call the function
    plot_salesman_routes(routes, coordinates, num_salesmen, dist)
"""