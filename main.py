import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
#
# from featuring import compute_atomic_number, compute_valence_number
# from utils import load_dataset
#
# tqdm.pandas() # bar progression when using pd.DataFrame.progress_apply
#
# from data.loader import load
#
# train = load(molecules_folder_path=r"./data/atoms/train",
#              energies_file_path=r"./data/energies/train.csv",
#              already_saved_file_path=r"./data/train.csv")
#
# test = load(molecules_folder_path=r"./data/atoms/test",
#             energies_file_path=None,
#             already_saved_file_path=r"./data/test.csv")
#
# ######################################
# # FEATURING : create new features here
# ######################################
#
# # atomic number
# train['atomic_number'] = train['atom_name'].progress_apply(lambda name: compute_atomic_number(name))
#
# # valence number
# train['valence_number'] = train['atomic_number'].progress_apply(lambda atomic_number: compute_valence_number(atomic_number))
#
# # number of atoms per molecule
# molecule_id_value_counts = train['molecule_id'].value_counts().to_dict()
# train['n_atoms'] = train['molecule_id'].progress_apply(lambda molecule_id: molecule_id_value_counts[molecule_id])
#
#
# # (pos_train, charges_train, valence_charges_train, energies_train) = load_dataset(data_path="./data", test=False)
# print("oui")
#
#
# # make a training algorithm on the data molecules to predict the energy of the molecule
#
# # Convert molecules and energies to numpy arrays
# X = train[['x', 'y', 'z']].to_numpy()  # Example feature (number of atoms)
# y = train['molecule_energy'].to_numpy().ravel()  # Target variable (energies)
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (pos_train, charges_train, valence_charges_train, energies_train) = load_dataset(data_path="./data", test=False)

#
# # make a training algorithm on the data molecules to predict the energy of the molecule
#
# # Convert molecules and energies to numpy arrays
# X = train[['x', 'y', 'z']].to_numpy()  # Example feature (number of atoms)
# y = train['molecule_energy'].to_numpy().ravel()  # Target variable (energies)
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Build the neural network model
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(None, 3)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))
#
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # Train the model
# model.fit(X_train_scaled, y_train, epochs=20, batch_size=100)
#
# # Evaluate the model
# train_score = model.evaluate(X_train_scaled, y_train)
# test_score = model.evaluate(X_test_scaled, y_test)
# y_pred = model.predict(X_test_scaled)
#
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE:", rmse)
