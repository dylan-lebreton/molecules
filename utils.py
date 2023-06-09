import os, numpy as np

ref_atoms = ['H', 'C', 'O', 'N', 'F', 'S', 'Cl']

def load_dataset(data_path="./data", test=False):
    if test:
        x_test = []
        full_path = os.path.join(data_path, "atoms", "test")
        for file_name in os.listdir(full_path):
            if file_name.endswith(".xyz"):
                file_path = os.path.join(full_path, file_name)
                id = int(file_name.split('_')[1].split('.')[0])
                with open(file_path, 'r') as file:
                    n = int(file.readline().replace("\n", ""))
                    pos = np.empty((n, 4))
                    lines = file.readlines()[1:]
                    for i in range(n):
                        a, x, y, z = lines[i].split()
                        pos[i,0] = ref_atoms.index(a)
                        pos[i,1:] = [float(p) for p in [x, y, z]]
                x_test.append((id, pos))
        return x_test
    else:
        x_train, y_train = [], []
        full_path_atoms = os.path.join(data_path, "atoms", "train")
        full_path_energies = os.path.join(data_path, "energies", "train.csv")
        with open(full_path_energies, 'r') as energies:
            for line_csv in energies.readlines()[1:]:
                id, energy = line_csv.split(',')
                file_path = os.path.join(full_path_atoms, f"id_{id}.xyz")
                with open(file_path, 'r') as file:
                    n = int(file.readline().replace("\n", ""))
                    pos = np.empty((n, 4))
                    lines = file.readlines()[1:]
                    for i in range(n):
                        a, x, y, z = lines[i].split()
                        pos[i,0] = ref_atoms.index(a)
                        pos[i,1:] = [float(p) for p in [x, y, z]]
                x_train.append((int(id), pos))
                y_train.append((int(id), float(energy)))
        return x_train, y_train
            
                