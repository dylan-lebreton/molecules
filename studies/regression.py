import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from data.loader import load_train_test

tqdm.pandas()

train, test = load_train_test(molecules_folder_path=r"../data/atoms/train",
                              energies_file_path=r"../data/energies/train.csv",
                              already_saved_file_path=r"../data/train.csv")

# number of atoms per molecule
for data in [train, test]:
    molecule_id_value_counts = data['molecule_id'].value_counts().to_dict()
    data['n_atoms'] = data['molecule_id'].progress_apply(lambda molecule_id: molecule_id_value_counts[molecule_id])

# show the correlation between the number of atoms and the energy of the molecule
train = train.drop_duplicates(subset=['molecule_id']).reset_index(drop=True).drop(columns=['atom_name', 'x', 'y', 'z'])

plt.figure()
plt.title("Number of atoms per molecule vs molecule energy")
plt.xlabel("Number of atoms per molecule")
plt.ylabel("Molecule energy")
plt.scatter(train['n_atoms'], train['molecule_energy'])
plt.show()


# we see that there is a correlation between the number of atoms and the energy of the molecule
# we can therefore use this feature to predict the energy of the molecule

# normalization of the data
scaler = StandardScaler()
train[['n_atoms_normalized']] = scaler.fit_transform(train[['n_atoms']])
test[['n_atoms_normalized']] = scaler.transform(test[['n_atoms']])

model = LinearRegression()
model.fit(train[['n_atoms_normalized']], train['molecule_energy'])
train['molecule_energy_prediction'] = model.predict(train[['n_atoms_normalized']])
test['molecule_energy_prediction'] = model.predict(test[['n_atoms_normalized']])

print('RMSE on train : ', np.sqrt(mean_squared_error(train['molecule_energy'], train['molecule_energy_prediction'])))
print('RMSE on test : ', np.sqrt(mean_squared_error(test['molecule_energy'], test['molecule_energy_prediction'])))