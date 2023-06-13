import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.loader import load

train = load(molecules_folder_path=r"./data/atoms/train",
             energies_file_path=r"./data/energies/train.csv",
             already_saved_file_path=r"./data/train.csv")

test = load(molecules_folder_path=r"./data/atoms/test",
            energies_file_path=None,
            already_saved_file_path=r"./data/test.csv")

# make a training algorithm on the data molecules to predict the energy of the molecule

# Convert molecules and energies to numpy arrays
X = train[['x', 'y', 'z']].to_numpy()  # Example feature (number of atoms)
y = train['molecule_energy'].to_numpy().ravel()  # Target variable (energies)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(None, 3)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=20, batch_size=100)

# Evaluate the model
train_score = model.evaluate(X_train_scaled, y_train)
test_score = model.evaluate(X_test_scaled, y_test)
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
