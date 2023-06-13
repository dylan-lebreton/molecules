import glob
import os
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

def molecule_id_from_molecule_file_path(molecule_file_path: Path) -> int:
    """
    Returns molecule's id from its file path.
    """
    return int(molecule_file_path.stem.split("_")[1])

def get_molecule_file_paths(molecules_folder_path: str) -> List[Path]:
    """
    Get's all molecules files paths from their folder path.
    """
    file_pattern = f'{molecules_folder_path}/*.xyz'
    file_names = glob.glob(file_pattern)
    return [Path(file_name) for file_name in file_names]


def energy_from_molecule_id(molecule_id: int, energies_file_path: Optional[str] = None) -> float:
    """
    Find molecule's energy in the energy file thanks to molecule's id.
    """
    if energies_file_path is None:
        result = np.nan
    else:
        energies = pd.read_csv(energies_file_path, sep=",")
        energy_row = energies.loc[energies['id'].astype(str) == str(molecule_id)]
        if len(energy_row) != 1:
            raise ValueError
        result = float(energy_row['energy'])
    return result

def lines_from_molecule_file_path(molecule_file_path: Path) -> List[str]:
    """
    Return the lines from the file defining a molecule except the two first ones which are useless.
    """
    with open(molecule_file_path, "r") as file:
        lines = file.readlines()
    return lines[2:]

def atom_data_from_molecule_file_path_line(line: str) -> Tuple[str, ...]:
    """
    Return a tuple of the data defining an atom (name, x, y, z) from a given line of the file defining the molecule.
    One line defines one atom.
    """
    result = list(line.strip().split())
    result[0] = str(result[0])
    for i in range(1, 3+1):
        result[i] = float(result[i])
    return tuple(result)

def load(molecules_folder_path: str, energies_file_path: Optional[str] = None,
         already_saved_file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Return pandas dataframe of the data given the location folder of the molecules files and the energy file.
    If the energy file is not provided (as for test set), the corresponding column in the dataframe will be fill with
    numpy.nan.

    If already_saved_file_path is given, the function tries to load the correspondant file (which should be a CSV)
    as dataframe.
    """

    # if already_save_file_path, we check if the CSV exist and load it
    if already_saved_file_path is not None and os.path.isfile(already_saved_file_path):
        result = pd.read_csv(already_saved_file_path, index_col=0)
    # else we remake the dataframe with all the data (takes more time)
    else:
        # iteration over files defining molecules
        result = list()
        for molecule_file_path in tqdm(get_molecule_file_paths(molecules_folder_path)):
            molecule_id = molecule_id_from_molecule_file_path(molecule_file_path)
            molecule_energy = energy_from_molecule_id(energies_file_path=energies_file_path,
                                                      molecule_id=molecule_id)
            # iteration over lines of the molecule file, each line defining an atom
            for line in lines_from_molecule_file_path(molecule_file_path):
                atom_name, x, y, z = atom_data_from_molecule_file_path_line(line)
                df = pd.DataFrame({"molecule_id": [molecule_id], "molecule_energy": [molecule_energy],
                                   "atom_name": [atom_name], "x": [x], "y": [y], "z": [z]})
                result.append(df)
        result = pd.concat(result, ignore_index=True).reset_index(drop=True)
    return result