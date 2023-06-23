"""

"""
from data.loader import load
from tools.featuring.scattering import load_scattering_coefficients

# retrieve of test data
train = load(molecules_folder_path=r"../data/atoms/test",
             energies_file_path=r"../data/energies/test.csv",
             already_saved_file_path=r"../data/test.csv")

# recuperation of scattering coefficients
train_scattering = load_scattering_coefficients(train['molecule_id'].drop_duplicates(),
                                                folder_path=r"../data/scattering",
                                                already_saved_file_path=r"../data/test_scattering.csv")

print("debug")


