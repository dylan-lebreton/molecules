from typing import List

import numpy as np
import pandas as pd
from PyAstronomy import pyasl
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


def rotate(x, y, z, u):
    return x*u, y*u, z*u

def translate(x, y, z, b):
    return x+b, y+b, z+b

def compute_atomic_number(atom_name: str) -> int:
    """
    Renvoie le numéro atomique de l'atome (nombre de protons dans le noyau).
    """
    an = pyasl.AtomicNo()
    return an.getAtomicNo(atom_name)

def compute_valence_number(atomic_number: int) -> int:
    """
    Renvoie le nombre d'électrons de la dernière couche de valence de l'atome (appelé nombre de valence).
    """
    if atomic_number <= 2:
        return atomic_number
    elif atomic_number <= 10:
        return atomic_number - 2
    elif atomic_number <= 18:
        return atomic_number - 10
    elif atomic_number <= 36:
        return atomic_number - 18
    elif atomic_number <= 54:
        return atomic_number - 36
    elif atomic_number <= 86:
        return atomic_number - 54
    else:
        raise ValueError("Atomic number is too large for this function.")

def atoms_distances_matrix(atoms_x_y_z_matrix: np.ndarray, two_dimensions: bool = True) -> np.ndarray:
    if two_dimensions:
        return squareform(pdist(atoms_x_y_z_matrix, metric='euclidean'))
    else:
        return pdist(atoms_x_y_z_matrix, metric='euclidean')

def split_array_in_batches(array: pd.DataFrame, batch_size: int) -> List[np.ndarray]:
    """
    Split an array into batches of size batch_size.
    """
    return [array[i:i+batch_size] for i in range(0, array.shape[0], batch_size)]

class PositionScaler:

    def __init__(self, overlapping_precision: float = 1e-1, sigma: float = 2.0):
        """
        The positions of the atoms are normalized so that two Gaussians of width sigma placed at these positions
        overlap with an amplitude less than overlapping_precision.

        This means that positions are adjusted so that when two Gaussians (representing two atoms) are overlapped,
        the amplitude of their overlap does not exceed a certain value defined by by overlapping_precision.

        This normalization is designed to ensure that the influence of two atoms in close proximity to each other
        is not disproportionate to that of more distant atoms.

        By limiting the overlap of Gaussians, we prevent the presence of two atoms very close together dominate
        the representation of the molecule in the scattering transform calculation.
        """
        self.overlapping_precision = overlapping_precision
        self.sigma = sigma
        self.delta = self.sigma * np.sqrt(-8 * np.log(self.overlapping_precision))
        self.molecules_distances_matrix = dict()
        self.minimal_atoms_distance = np.nan

    def compute_molecules_distances_matrix(self, dataframe: pd.DataFrame, molecule_id_column, x_column, y_column, z_column):

        # iteration over molecules
        tqdm_desc = "Position Scaler - computation of molecules distances matrix"
        for molecule_id in tqdm(dataframe[molecule_id_column].unique(), desc=tqdm_desc):

            # sub dataframe of the molecule
            sub = dataframe[dataframe[molecule_id_column] == molecule_id]

            # matrix of x, y, z coordinates of atoms in the molecule
            x_y_z_matrix = sub[[x_column, y_column, z_column]].to_numpy()

            # computation of 1D distances matrix
            distances_matrix = atoms_distances_matrix(x_y_z_matrix, two_dimensions=False)

            self.molecules_distances_matrix[molecule_id] = distances_matrix

    def compute_minimal_atoms_distance(self):
        if len(self.molecules_distances_matrix) == 0:
            raise ValueError("Molecules distances matrix has not been computed yet.")
        else:
            self.minimal_atoms_distance =  min([np.min(distances_matrix) for distances_matrix in \
                                                self.molecules_distances_matrix.values()])

    def fit(self, dataframe: pd.DataFrame, molecule_id_column, x_column, y_column, z_column):

        self.compute_molecules_distances_matrix(dataframe, molecule_id_column, x_column, y_column, z_column)
        self.compute_minimal_atoms_distance()

    def transform(self, dataframe: pd.DataFrame,
                  x_column, x_rescaled_column,
                  y_column, y_rescaled_column,
                  z_column, z_rescaled_column):

        if np.isnan(self.minimal_atoms_distance):
            raise ValueError("Minimal atoms distance has not been computed yet.")

        dataframe[x_rescaled_column]  = dataframe[x_column] * self.delta / self.minimal_atoms_distance
        dataframe[y_rescaled_column]  = dataframe[y_column] * self.delta / self.minimal_atoms_distance
        dataframe[z_rescaled_column]  = dataframe[z_column] * self.delta / self.minimal_atoms_distance

    def fit_transform(self, dataframe: pd.DataFrame, molecule_id_column,
                      x_column, x_rescaled_column,
                      y_column, y_rescaled_column,
                      z_column, z_rescaled_column):

        self.fit(dataframe, molecule_id_column, x_column, y_column, z_column)
        self.transform(dataframe, x_column, x_rescaled_column, y_column, y_rescaled_column, z_column, z_rescaled_column)

def batch_attribution(molecule_id_series, batch_size):
    """
    This function attributes a batch number to each molecule id (grouping molecules by batch of size batch_size).
    """
    # Create a dictionary where keys are unique molecule ids
    # and values are the batch number they belong to
    molecule_ids = molecule_id_series.unique()
    batches = len(molecule_ids) // batch_size + (len(molecule_ids) % batch_size > 0)

    # initialize an empty dictionary
    molecule_to_batch = {}

    for batch_id in range(batches):
        # get the molecule ids for this batch
        batch_molecule_ids = molecule_ids[batch_id * batch_size:(batch_id + 1) * batch_size]

        # add each molecule id to the dictionary with the batch_id as the value
        for molecule_id in batch_molecule_ids:
            molecule_to_batch[molecule_id] = batch_id

    return molecule_id_series.map(molecule_to_batch)


def position_tensor_from_dataframe(dataframe: pd.DataFrame, molecule_id_column,
                                   x_column, y_column, z_column, max_number_of_atoms):
    """
    Returns a tensor of positions of the atoms of the molecule.
    """

    # retrieve molecules ids
    molecules_ids = dataframe[molecule_id_column].unique()

    # initialize the position tensor (to zeros ==> zero padding)
    position_tensor = np.zeros((len(molecules_ids), max_number_of_atoms, 3))

    # iteration over molecules
    for molecule_id_index in range(len(molecules_ids)):

        # retrieve part of the dataframe corresponding to the molecule
        molecule_id = molecules_ids[molecule_id_index]
        sub = dataframe[dataframe[molecule_id_column] == molecule_id].copy(deep=True).reset_index(drop=True)

        # iteration over atoms
        for index, row in sub.iterrows():
            position_tensor[molecule_id_index, index, 0] = row[x_column]
            position_tensor[molecule_id_index, index, 1] = row[y_column]
            position_tensor[molecule_id_index, index, 2] = row[z_column]

    return position_tensor

def charges_tensor_from_dataframe(dataframe: pd.DataFrame, molecule_id_column, charges_column, max_number_of_atoms):

    # retrieve molecules ids
    molecules_ids = dataframe[molecule_id_column].unique()

    # initialize the tensor (to zeros ==> zero padding)
    tensor = np.zeros((len(molecules_ids), max_number_of_atoms))

    # iteration over molecules
    for molecule_id_index in range(len(molecules_ids)):

        # retrieve part of the dataframe corresponding to the molecule
        molecule_id = molecules_ids[molecule_id_index]
        sub = dataframe[dataframe[molecule_id_column] == molecule_id].copy(deep=True).reset_index(drop=True)

        # iteration over atoms
        for index, row in sub.iterrows():
            tensor[molecule_id_index, index] = row[charges_column]

    return tensor