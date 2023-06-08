import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

#faire un fichier utils.py pour trier les données : ordre croissant des coordonnées x



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
            
    
    def Permutation_symmetry(self,sigma): #à refaire
        loc=0
        for Permutation in sigma:
            self.atoms[Permutation]=self.atoms[loc]
            loc+=1
            

# Create an example molecule
molecule = Molecule()

atom1 = Atom("C", 0.0, 0.0, 0.0)
atom2 = Atom("H", 1.0, 0.0, 0.0)
atom3 = Atom("H", 0.0, 1.0, 0.0)
atom4 = Atom("H", 0.0, 0.0, 1.0)

molecule.add_atom(atom1)
molecule.add_atom(atom2)
molecule.add_atom(atom3)
molecule.add_atom(atom4)

#molecule.Translational_symmetry(2) 
#molecule.Rotational_symmetry(2)
#print('Avant permutation')
molecule.print_atoms()
#molecule.Permutation_symmetry([3,2,1,0])
#print('Apres permutation')
#molecule.print_atoms()

    
    