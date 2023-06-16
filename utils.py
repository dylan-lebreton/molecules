import os, numpy as np
from PyAstronomy import pyasl

an = pyasl.AtomicNo()

# Maximum number of atoms in one molecule in the dataset
N_MAX = 23

def load_scattering(train_file_0="order_0_train_qm7.npy", train_file_1_2="orders_1_and_2_train_qm7.npy", 
                    test_file_0="order_0_test_qm7.npy", test_file_1_2="orders_1_and_2_test_qm7.npy",
                    folder="data"):
    order_0_train = np.load(os.path.join(folder, train_file_0))
    order_0_test = np.load(os.path.join(folder, test_file_0))
    orders_1_2_train = np.load(os.path.join(folder, train_file_1_2))
    orders_1_2_test = np.load(os.path.join(folder, test_file_1_2))
    scattering_coef_train = np.concatenate([order_0_train, orders_1_2_train], axis=1)
    scattering_coef_test = np.concatenate([order_0_test, orders_1_2_test], axis=1)
    return scattering_coef_train, scattering_coef_test
    

def load_dataset(data_path="./data", test=False):
    # Load Test data to predict
    if test:
        # List all files in directory, filter only "xyz" ones and sort by id
        full_path = os.path.join(data_path, "atoms", "test")
        files = os.listdir(full_path)
        filter(lambda x: x.endswith(".xyz"), files)
        files.sort(key=lambda file_name:int(file_name.split('_')[1].split('.')[0]))
        # Get number of molecules in test sample
        n_molecules = len(files)
        # Pre-allocate tabs for data
        pos_test = np.zeros((n_molecules, N_MAX, 3), dtype=np.float32)
        charges_test = np.zeros((n_molecules, N_MAX), dtype=np.int32)
        # Iterate over every file/molecule
        for k, file_name in enumerate(files):
            file_path = os.path.join(full_path, file_name)
            with open(file_path, 'r') as file:
                # Get number of atoms in molecule
                n = int(file.readline().replace("\n", ""))
                assert n <= N_MAX, "Too many atoms in molecule"
                lines = file.readlines()[1:]
                # Iterate over every atoms in molecule
                for i in range(n):
                    a, x, y, z = lines[i].split()
                    # Get atomic number/number of charges in atom
                    charges_test[k, i] = an.getAtomicNo(a)
                    # Get positional data
                    for j, p in enumerate([x, y, z]):
                        pos_test[k, i, j] = float(p)
        # Get valence charge from full charge data 
        mask = charges_test <= 2
        valence_charges_test = charges_test * mask

        mask = np.logical_and(charges_test > 2, charges_test <= 10)
        valence_charges_test += (charges_test - 2) * mask

        mask = np.logical_and(charges_test > 10, charges_test <= 18)
        valence_charges_test += (charges_test - 10) * mask
        return (pos_test, charges_test, valence_charges_test)
    # Load train data and energy for training
    else:
        full_path_atoms = os.path.join(data_path, "atoms", "train")
        full_path_energies = os.path.join(data_path, "energies", "train.csv")
        with open(full_path_energies, 'r') as energies:
            csv_lines = energies.readlines()[1:]
            # Get number of molecules in train sample
            n_molecules = len(csv_lines)
            # Pre-allocate tabs for data
            pos_train = np.zeros((n_molecules, N_MAX, 3), dtype=np.float32)
            charges_train = np.zeros((n_molecules, N_MAX), dtype=np.int32)
            energies_train = np.zeros((n_molecules), dtype=np.float32)
            # Iterate over every file/molecule
            for csv_line in csv_lines:
                id, energy = csv_line.split(',')
                id = int(id)
                file_path = os.path.join(full_path_atoms, f"id_{id}.xyz")
                with open(file_path, 'r') as file:
                    # Get number of atoms in molecule
                    n = int(file.readline().replace("\n", ""))
                    assert n <= N_MAX, "Too many atoms in molecule"
                    lines = file.readlines()[1:]
                    # Iterate over every atoms in molecule
                    for i in range(n):
                        a, x, y, z = lines[i].split()
                        # Get atomic number/number of charges in atom
                        charges_train[id-1, i] = an.getAtomicNo(a)
                        # Get positional data
                        for j, p in enumerate([x, y, z]):
                            pos_train[id-1, i, j] = float(p)
                    # Get energy
                    energies_train[id-1] = float(energy)
        # Get valence charge from full charge data 
        mask = charges_train <= 2
        valence_charges_train = charges_train * mask

        mask = np.logical_and(charges_train > 2, charges_train <= 10)
        valence_charges_train += (charges_train - 2) * mask

        mask = np.logical_and(charges_train > 10, charges_train <= 18)
        valence_charges_train += (charges_train - 10) * mask
        return (pos_train, charges_train, valence_charges_train, energies_train)