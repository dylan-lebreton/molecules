import numpy as np
from scipy.spatial.distance import squareform, pdist
from spektral.data import Graph


def generate_graph(molecule_dataframe, x_column, y_column, z_column, charges_column=None, valence_column=None,
                   molecule_energy_column=None):
    # we define the columns that will be used for the nodes features
    columns_for_nodes_features = [x_column, y_column, z_column]
    if charges_column is not None:
        columns_for_nodes_features.append(charges_column)
    if valence_column is not None:
        columns_for_nodes_features.append(valence_column)

    # set the nodes features
    nodes = molecule_dataframe[columns_for_nodes_features].values
    nodes = nodes.astype("float32")

    # calculate adjacency matrix based on Euclidean distance
    positions = molecule_dataframe[[x_column, y_column, z_column]].values
    adjacency = squareform(pdist(positions))
    adjacency = adjacency.astype('float32')

    # define the graph object
    if molecule_energy_column is not None:
        result = Graph(nodes, adjacency, y=molecule_dataframe[molecule_energy_column].values[0])
    else:
        result = Graph(nodes, adjacency, y=np.nan)

    return result
