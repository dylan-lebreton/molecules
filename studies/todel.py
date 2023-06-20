from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from data.loader import load
from tools.featuring import compute_atomic_number, compute_valence_number

tqdm.pandas()

data = load(molecules_folder_path=r"../data/atoms/train",
            energies_file_path=r"../data/energies/train.csv",
            already_saved_file_path=r"../data/train.csv")

# # atomic number
# data['atomic_number'] = data['atom_name'].progress_apply(lambda name: compute_atomic_number(name))
#
# # valence number
# data['valence_number'] = data['atomic_number'].progress_apply(
#     lambda atomic_number: compute_valence_number(atomic_number))

for molecule_id in data.molecule_id.unique():
    df = data.loc[data['molecule_id'] == molecule_id]

    adjacency_matrix = squareform(pdist(df[['x', 'y', 'z']].values))

    print("oui")
# # Generate adjacency matrix based on Euclidean distance
# coordinates = molecule_dataframe[['x', 'y', 'z']].values
# adjacency_matrix = squareform(pdist(coordinates))
#
# print("debug")
