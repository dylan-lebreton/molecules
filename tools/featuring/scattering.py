import os

import numpy as np
import torch
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D
from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians
from kymatio.torch import HarmonicScattering3D
from tqdm import tqdm

from data.loader import load
from tools.featuring.basic import compute_atomic_number, compute_valence_number, PositionScaler, batch_attribution, \
    position_tensor_from_dataframe, charges_tensor_from_dataframe

# noinspection DuplicatedCode
# retrieve of train data
train = load(molecules_folder_path=r"../data/atoms/train",
             energies_file_path=r"../data/energies/train.csv",
             already_saved_file_path=r"./train.csv")

# computation of atomic number
tqdm.pandas(desc="Computing Atomic Number")
train['Z'] = train['atom_name'].progress_apply(lambda name: compute_atomic_number(name))

# sort the dataframe by molecule_id and atome atomic number
train.sort_values(by=['molecule_id', 'Z'], inplace=True)

# computation of valence number
tqdm.pandas(desc="Computing Valence Number")
train['valence_number'] = train['Z'].progress_apply(lambda atomic_number: compute_valence_number(atomic_number))

# computation of the number of atoms in each molecule
train['n_atoms'] = train.groupby('molecule_id')['molecule_id'].transform('count')

# rescaled of x, y, and z position of atoms based on the minimal atoms distance computed over all molecules
position_scaler = PositionScaler(overlapping_precision=1e-1, sigma=2.0)
position_scaler.fit_transform(train, molecule_id_column='molecule_id',
                              x_column='x', x_rescaled_column="x_rescaled",
                              y_column='y', y_rescaled_column="y_rescaled",
                              z_column='z', z_rescaled_column="z_rescaled")

# attribute batch number to each molecule
train['batch'] = batch_attribution(train['molecule_id'], batch_size=30)

# definition of 3D grid
# noinspection DuplicatedCode
M, N, O = 192, 128, 96
grid = np.mgrid[-M // 2:-M // 2 + M, -N // 2:-N // 2 + N, -O // 2:-O // 2 + O]
grid = np.fft.ifftshift(grid)

# definition of the scattering transform
# J est le nombre maximum d'ondelettes utilisées
# L est le nombre de directions pour les ondelettes
# integral_powers est la liste des exposants utilisés pour calculer l'intégrale de la densité de charge
sigma = 2.0
integral_powers = [0.5, 1.0, 2.0, 3.0]
scattering = HarmonicScattering3D(J=2, shape=(M, N, O), L=3, sigma_0=sigma, integral_powers=integral_powers)

# set the device to cuda if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
scattering.to(device)

# iteration over each batch of molecules
for batch_number in tqdm(train['batch'].unique(), desc="Computing Scattering Coefficients"):

    # retrieve the batch
    batch = train[train['batch'] == batch_number]

    # compute the position tensor
    positions = position_tensor_from_dataframe(dataframe=batch, molecule_id_column="molecule_id",
                                               x_column="x_rescaled", y_column="y_rescaled", z_column="z_rescaled",
                                               max_number_of_atoms=train['n_atoms'].max())

    # compute the atomic number tensor
    charges = charges_tensor_from_dataframe(dataframe=batch, molecule_id_column="molecule_id",
                                            charges_column="Z", max_number_of_atoms=train['n_atoms'].max())

    # compute the valence number tensor
    valences = charges_tensor_from_dataframe(dataframe=batch, molecule_id_column="molecule_id",
                                             charges_column="valence_number",
                                             max_number_of_atoms=train['n_atoms'].max())

    # calculate the density map for the atomic number charges and transfer to PyTorch
    charges_density_map = generate_weighted_sum_of_gaussians(grid, positions, charges, sigma)
    charges_density_map = torch.from_numpy(charges_density_map)
    charges_density_map = charges_density_map.to(device).float()
    # compute zeroth-order, first-order, and second-order scattering coefficients of the atomic number
    charges_zero_order_scattering_coefs = TorchBackend3D.compute_integrals(charges_density_map, integral_powers)
    charges_first_sec_order_scattering_coefs = scattering(charges_density_map)

    # calculate the density map for the valence number charges and transfer to PyTorch
    valences_density_map = generate_weighted_sum_of_gaussians(grid, positions, valences, sigma)
    valences_density_map = torch.from_numpy(valences_density_map)
    valences_density_map = valences_density_map.to(device).float()
    # compute zeroth-order, first-order, and second-order scattering coefficients of the valence charges
    valences_zero_order_scattering_coefs = TorchBackend3D.compute_integrals(valences_density_map, integral_powers)
    valences_first_sec_order_scattering_coefs = scattering(valences_density_map)

    # take the difference between nuclear and valence charges, then compute the corresponding scattering coefficients
    core_density_map = charges_density_map - valences_density_map
    core_zero_order_scattering_coefs = TorchBackend3D.compute_integrals(core_density_map, integral_powers)
    core_first_sec_order_scattering_coefs = scattering(core_density_map)

    # we now save all scattering coefficients in a folder
    for i, molecule_id in enumerate(batch['molecule_id'].unique()):
        np.save(f"../../data/scattering{molecule_id}_charges_zero.npy",
                charges_zero_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_charges_first_sec.npy",
                charges_first_sec_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_valences_zero.npy",
                valences_zero_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_valences_first_sec.npy",
                valences_first_sec_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_core_zero.npy",
                core_zero_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_core_first_sec.npy",
                core_first_sec_order_scattering_coefs.cpu().numpy()[i, ...])

# Now we do the same on test data, but using PositionScaler already fitted on train data
test = load(molecules_folder_path=r"../data/atoms/test",
            energies_file_path=r"../data/energies/test.csv",
            already_saved_file_path=r"./test.csv")

# computation of atomic number
tqdm.pandas(desc="Computing Atomic Number")
test['Z'] = test['atom_name'].progress_apply(lambda name: compute_atomic_number(name))

# sort the dataframe by molecule_id and atome atomic number
test.sort_values(by=['molecule_id', 'Z'], inplace=True)

# computation of valence number
tqdm.pandas(desc="Computing Valence Number")
test['valence_number'] = test['Z'].progress_apply(lambda atomic_number: compute_valence_number(atomic_number))

# computation of the number of atoms in each molecule
test['n_atoms'] = test.groupby('molecule_id')['molecule_id'].transform('count')

# rescaled of x, y, and z position of atoms based on the minimal atoms distance computed over all train molecules
position_scaler.transform(test,
                          x_column='x', x_rescaled_column="x_rescaled",
                          y_column='y', y_rescaled_column="y_rescaled",
                          z_column='z', z_rescaled_column="z_rescaled")

# attribute batch number to each molecule
test['batch'] = batch_attribution(test['molecule_id'], batch_size=30)


# iteration over each batch of molecules
for batch_number in tqdm(test['batch'].unique(), desc="Computing Scattering Coefficients"):

    # retrieve the batch
    batch = test[test['batch'] == batch_number]

    # compute the position tensor
    positions = position_tensor_from_dataframe(dataframe=batch, molecule_id_column="molecule_id",
                                               x_column="x_rescaled", y_column="y_rescaled", z_column="z_rescaled",
                                               max_number_of_atoms=train['n_atoms'].max())

    # compute the atomic number tensor
    charges = charges_tensor_from_dataframe(dataframe=batch, molecule_id_column="molecule_id",
                                            charges_column="Z", max_number_of_atoms=train['n_atoms'].max())

    # compute the valence number tensor
    valences = charges_tensor_from_dataframe(dataframe=batch, molecule_id_column="molecule_id",
                                             charges_column="valence_number",
                                             max_number_of_atoms=train['n_atoms'].max())

    # calculate the density map for the atomic number charges and transfer to PyTorch
    charges_density_map = generate_weighted_sum_of_gaussians(grid, positions, charges, sigma)
    charges_density_map = torch.from_numpy(charges_density_map)
    charges_density_map = charges_density_map.to(device).float()
    # compute zeroth-order, first-order, and second-order scattering coefficients of the atomic number
    charges_zero_order_scattering_coefs = TorchBackend3D.compute_integrals(charges_density_map, integral_powers)
    charges_first_sec_order_scattering_coefs = scattering(charges_density_map)

    # calculate the density map for the valence number charges and transfer to PyTorch
    valences_density_map = generate_weighted_sum_of_gaussians(grid, positions, valences, sigma)
    valences_density_map = torch.from_numpy(valences_density_map)
    valences_density_map = valences_density_map.to(device).float()
    # compute zeroth-order, first-order, and second-order scattering coefficients of the valence charges
    valences_zero_order_scattering_coefs = TorchBackend3D.compute_integrals(valences_density_map, integral_powers)
    valences_first_sec_order_scattering_coefs = scattering(valences_density_map)

    # take the difference between nuclear and valence charges, then compute the corresponding scattering coefficients
    core_density_map = charges_density_map - valences_density_map
    core_zero_order_scattering_coefs = TorchBackend3D.compute_integrals(core_density_map, integral_powers)
    core_first_sec_order_scattering_coefs = scattering(core_density_map)

    # we now save all scattering coefficients in a folder
    for i, molecule_id in enumerate(batch['molecule_id'].unique()):
        np.save(f"../../data/scattering{molecule_id}_charges_zero.npy",
                charges_zero_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_charges_first_sec.npy",
                charges_first_sec_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_valences_zero.npy",
                valences_zero_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_valences_first_sec.npy",
                valences_first_sec_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_core_zero.npy",
                core_zero_order_scattering_coefs.cpu().numpy()[i, ...])
        np.save(f"../../data/scattering{molecule_id}_core_first_sec.npy",
                core_first_sec_order_scattering_coefs.cpu().numpy()[i, ...])

"""
En chaque point de la grille, on calcule la densité de charge due à la présence des atomes de la molécule. 
Chaque atome est représenté par une fonction gaussienne (une courbe en forme de cloche) centrée sur sa position, 
et pondérée par sa charge atomique.

Lorsqu'on "place" ces gaussiennes sur la grille, on obtient une représentation tridimensionnelle de la 
densité de charge de la molécule : en chaque point de la grille, la valeur calculée est la somme des valeurs des 
gaussiennes pour toutes les positions atomiques. Ainsi, la valeur en un point donné de la grille est élevée 
s'il est proche d'un ou plusieurs atomes, et faible s'il est éloigné de tous les atomes.

C'est cette densité de charge qui est ensuite utilisée pour calculer la transformée scattering, 
qui est une représentation qui conserve certaines propriétés de symétrie et qui est utilisée pour prédire 
l'énergie de la molécule.
"""
