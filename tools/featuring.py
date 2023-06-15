from PyAstronomy import pyasl

def rotate(x, y, z, u):
    return x*u, y*u, z*u

def translate(x, y, z, b):
    return x+b, y+b, z+b

def compute_atomic_number(atom_name: str) -> int:
    an = pyasl.AtomicNo()
    return an.getAtomicNo(atom_name)

def compute_valence_number(atomic_number: int) -> int:
    """
    Renvoie le nombre d'électrons de la dernière couche de valence de l'atome (appelé nombre de valence).
    """
    if atomic_number <= 2:
        return atomic_number
    elif atomic_number <= 10:
        return atomic_number - 2
    elif atomic_number <= 18:
        return atomic_number - 10
    elif atomic_number <= 36:
        return atomic_number - 18
    elif atomic_number <= 54:
        return atomic_number - 36
    elif atomic_number <= 86:
        return atomic_number - 54
    else:
        raise ValueError("Atomic number is too large for this function.")