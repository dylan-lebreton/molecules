"""
Regression on the number of atoms per molecule
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data.loader import load_train_test, load
from tools.io import to_kaggle_csv

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

# make prediction on real test data
test = load(molecules_folder_path=r"../data/atoms/test",
            energies_file_path=r"../data/energies/test.csv",
            already_saved_file_path=r"../data/test.csv")

# computation of the number of atoms per molecule for the real test data
test_molecule_id_value_counts = test['molecule_id'].value_counts().to_dict()
test['n_atoms'] = test['molecule_id'].progress_apply(lambda molecule_id: test_molecule_id_value_counts[molecule_id])

# keep only per molecule
test = test.drop_duplicates(subset=['molecule_id']).reset_index(drop=True).drop(columns=['atom_name', 'x', 'y', 'z'])

test[['n_atoms_normalized']] = scaler.transform(test[['n_atoms']])
test['molecule_energy'] = model.predict(test[['n_atoms_normalized']])
to_kaggle_csv(test, csv_file_path=r"../data/kaggle/regression.csv",
              molecule_id_column='molecule_id',
              molecule_prediction_column='molecule_energy')
