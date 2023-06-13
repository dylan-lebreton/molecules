import numpy as np
import glob
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

#faire un fichier utils.py pour trier les données : ordre croissant des coordonnées x
#Pour l'instant l'import se fait via ce code, je ne sais pas si c'est la meilleure solution.

class Atom:
    def __init__(self, name, x, y, z):
        self.name = name
        self.x = x
        self.y = y
        self.z = z

class Molecule:
    def __init__(self):
        self.atoms = []

    def add_atom(self, atom):
        self.atoms.append(atom)

    def get_num_atoms(self):
        return len(self.atoms)

    def print_atoms(self):
        for atom in self.atoms:
            print("Atom:", atom.name)
            print("Position (x, y, z):", atom.x, atom.y, atom.z)
            
    def sort_atoms(self):
        self.atoms.sort(key=lambda atom: (atom.x, atom.y, atom.z))
    
    def Translational_symmetry(self,b):
        for atom in self.atoms:
            atom.x += b
            atom.y += b
            atom.z += b
            
    def Rotational_symmetry(self,U):
        for atom in self.atoms:
            atom.x *= U
            atom.y *= U
            atom.z *= U
                        
# import the data from the file 'Atoms/train' 

molecule = Molecule()

molecules = [] # list of molecules

file_pattern = 'data/atoms/train/*.xyz'
file_names = glob.glob(file_pattern)

for file_name in file_names:
    molecule = Molecule()
    
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines[2:]:  # skip the first two lines
            atom_info = line.strip().split() # remove whitespace and split on comma
            name = atom_info[0] 
            x = float(atom_info[1])
            y = float(atom_info[2])
            z = float(atom_info[3])
            atom = Atom(name, x, y, z) 
            molecule.add_atom(atom) 

    # Add the molecule to the list
    molecules.append(molecule)
    
for molecule in molecules:
    molecule.sort_atoms()

#import the energies in a list

energies = [] # list of energies
with open('data/energies/train.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        energy = float(row[1])
        energies.append(energy)

#print(energies)
    
#molecules[0].print_atoms()
#print(molecules[0].atoms[0].x)    # coordinate x exemple
#print(molecules[0].atoms[0].name)   # name of the atom exemple

#make a training algorithm on the data molecules to predict the energy of the molecule

# Convert molecules and energies to numpy arrays
X = np.array([molecule.get_num_atoms() for molecule in molecules])  # Example feature (number of atoms)
y = np.array(energies)  # Target variable (energies)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
X_test_scaled = scaler.transform(X_test.reshape(-1, 1))

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))
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