import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.loader import load

# matplotlib.use('TkAgg')
tqdm.pandas()

train = load(molecules_folder_path=r"../data/atoms/train",
             energies_file_path=r"../data/energies/train.csv",
             already_saved_file_path=r"../data/train.csv")

# number of atoms per molecule
molecule_id_value_counts = train['molecule_id'].value_counts().to_dict()
train['n_atoms'] = train['molecule_id'].progress_apply(lambda molecule_id: molecule_id_value_counts[molecule_id])

# keep only molecule level information
train = train.drop_duplicates(subset=['molecule_id']).reset_index(drop=True).drop(columns=['atom_name', 'x', 'y', 'z'])

plt.figure()
plt.title("Number of atoms per molecule vs molecule energy")
plt.xlabel("Number of atoms per molecule")
plt.ylabel("Molecule energy")
plt.scatter(train['n_atoms'], train['molecule_energy'])
plt.show()

print("oui")
