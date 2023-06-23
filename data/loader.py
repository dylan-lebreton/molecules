import glob
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
    for i in range(1, 3 + 1):
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
        for molecule_file_path in tqdm(get_molecule_file_paths(molecules_folder_path), desc="Loading data"):
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


def load_train_val_test(molecules_folder_path: str, energies_file_path: Optional[str] = None,
                        already_saved_file_path: Optional[str] = None,
                        train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                        random_state: int = 42) -> Tuple[any, any, any]:
    data = load(molecules_folder_path=molecules_folder_path, energies_file_path=energies_file_path,
                already_saved_file_path=already_saved_file_path)

    # retrieve molecules ids
    molecule_ids = data['molecule_id'].drop_duplicates().to_list()

    # split the molecules ids in train and (val, test)
    train_ids, val_ids, test_ids = split_array_in_train_val_test(array=molecule_ids,
                                                                 train_ratio=train_ratio,
                                                                 val_ratio=val_ratio,
                                                                 test_ratio=test_ratio,
                                                                 random_state=random_state)

    # define dataframes with previous ids
    train = data.loc[data['molecule_id'].isin(train_ids)].copy(deep=True).reset_index(drop=True)
    val = data.loc[data['molecule_id'].isin(val_ids)].copy(deep=True).reset_index(drop=True)
    test = data.loc[data['molecule_id'].isin(test_ids)].copy(deep=True).reset_index(drop=True)

    return train, val, test


def load_train_test(molecules_folder_path: str, energies_file_path: Optional[str] = None,
                    already_saved_file_path: Optional[str] = None,
                    train_ratio: float = 0.8, test_ratio: float = 0.2,
                    random_state: int = 42) -> Tuple[any, any]:
    data = load(molecules_folder_path=molecules_folder_path, energies_file_path=energies_file_path,
                already_saved_file_path=already_saved_file_path)

    # retrieve molecules ids
    molecule_ids = data['molecule_id'].drop_duplicates().to_list()

    # split the molecules ids in train and test
    train_ids, test_ids = split_array_in_train_test(array=molecule_ids, train_ratio=train_ratio,
                                                    test_ratio=test_ratio, random_state=random_state)

    # define dataframes with previous ids
    train = data.loc[data['molecule_id'].isin(train_ids)].copy(deep=True).reset_index(drop=True)
    test = data.loc[data['molecule_id'].isin(test_ids)].copy(deep=True).reset_index(drop=True)

    return train, test


def split_array_in_train_test(array, train_ratio: float, test_ratio: float, random_state: int = 42):
    assert train_ratio + test_ratio == 1.0

    # split in train and test data
    train, test = train_test_split(array, test_size=test_ratio, random_state=random_state)

    return train, test


def split_array_in_train_val_test(array, train_ratio: float, val_ratio: float, test_ratio: float,
                                  random_state: int = 42):
    assert train_ratio + val_ratio + test_ratio == 1.0

    # split in train and (val, test) data
    train, val_and_test = train_test_split(array, test_size=val_ratio + test_ratio, random_state=random_state)

    # computation of test_ratio on the remaining data
    remaining_test_ratio = test_ratio / (val_ratio + test_ratio)

    # split (val, test) in val and test
    val, test = train_test_split(val_and_test, test_size=remaining_test_ratio, random_state=random_state)

    return train, val, test
