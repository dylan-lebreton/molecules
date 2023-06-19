import numpy as np
from scipy.spatial.distance import pdist
import os
from tqdm import trange

from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend \
    import TorchBackend3D
from kymatio.scattering3d.utils \
    import generate_weighted_sum_of_gaussians

from sklearn import linear_model, model_selection, preprocessing, pipeline
from scipy.spatial.distance import pdist

from utils import load_dataset, load_scattering
import argparse
import torch

def normalize_pos(pos, full_charges, sigma = 2.0, overlapping_precision = 1e-1):
    """
    We normalize the positions of the atoms. 
    Specifically, the positions are rescaled such that two Gaussians 
    of width sigma placed at those positions overlap with amplitude 
    less than overlapping_precision.
    """
    overlapping_precision = 1e-1
    sigma = 2.0
    min_dist = np.inf
    n_molecules = pos.shape[0]
    for i in range(n_molecules):
        n_atoms = np.sum(full_charges[i] != 0)
        pos_i = pos[i, :n_atoms, :]
        min_dist = min(min_dist, pdist(pos_i).min())

    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
    pos = pos * delta / min_dist
    return pos

def scattering(pos, full_charges, valence_charges, ident, batch_size=8, J=2, M=192,
               N=128, O=96, L=3, sigma=2.0,
               integral_powers = [0.5, 1.0, 2.0, 3.0],
               save_to_file = True):
    """Code for scattering taken from kymatio tutorial on 3D scattering.
    https://www.kymat.io/gallery_3d/scattering3d_qm7_torch.html
    """
    grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O]
    grid = np.fft.ifftshift(grid)
    scattering = HarmonicScattering3D(J=J, shape=(M, N, O),
                                  L=L, sigma_0=sigma,
                                  integral_powers=integral_powers)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scattering.to(device)

    n_molecules = pos.shape[0]

    n_batches = int(np.ceil(n_molecules / batch_size))
    order_0, orders_1_and_2 = [], []
    print('Computing solid harmonic scattering coefficients of '
        '{} molecules from the QM7 database on {}'.format(
            n_molecules,   "GPU" if use_cuda else "CPU"))
    print('sigma: {}, L: {}, J: {}, integral powers: {}'.format(
            sigma, L, J, integral_powers))

    this_time = None
    last_time = None
    for i in trange(n_batches, desc="Scattering"):
        # Extract the current batch.
        start = i * batch_size
        end = min(start + batch_size, n_molecules)

        pos_batch = pos[start:end]
        full_batch = full_charges[start:end]
        val_batch = valence_charges[start:end]

        # Calculate the density map for the nuclear charges and transfer
        # to PyTorch.
        full_density_batch = generate_weighted_sum_of_gaussians(grid,
                pos_batch, full_batch, sigma)
        full_density_batch = torch.from_numpy(full_density_batch)
        full_density_batch = full_density_batch.to(device).float()

        # Compute zeroth-order, first-order, and second-order scattering
        # coefficients of the nuclear charges.
        full_order_0 = TorchBackend3D.compute_integrals(full_density_batch,
                                        integral_powers)
        full_scattering = scattering(full_density_batch)

        # Compute the map for valence charges.
        val_density_batch = generate_weighted_sum_of_gaussians(grid,
                pos_batch, val_batch, sigma)
        val_density_batch = torch.from_numpy(val_density_batch)
        val_density_batch = val_density_batch.to(device).float()

        # Compute scattering coefficients for the valence charges.
        val_order_0 = TorchBackend3D.compute_integrals(val_density_batch,
                                        integral_powers)
        val_scattering = scattering(val_density_batch)

        # Take the difference between nuclear and valence charges, then
        # compute the corresponding scattering coefficients.
        core_density_batch = full_density_batch - val_density_batch

        core_order_0 = TorchBackend3D.compute_integrals(core_density_batch,
                                        integral_powers)
        core_scattering = scattering(core_density_batch)

        # Stack the nuclear, valence, and core coefficients into arrays
        # and append them to the output.
        batch_order_0 = torch.stack(
            (full_order_0, val_order_0, core_order_0), dim=-1)
        batch_orders_1_and_2 = torch.stack(
            (full_scattering, val_scattering, core_scattering), dim=-1)

        order_0.append(batch_order_0)
        orders_1_and_2.append(batch_orders_1_and_2)

    order_0 = torch.cat(order_0, dim=0)
    orders_1_and_2 = torch.cat(orders_1_and_2, dim=0)

    order_0 = order_0.cpu().numpy()
    orders_1_and_2 = orders_1_and_2.cpu().numpy()

    order_0 = order_0.reshape((n_molecules, -1))
    orders_1_and_2 = orders_1_and_2.reshape((n_molecules, -1))

    if save_to_file:
        basename = f'{ident}_qm7.npy'

        cache_dir = "scattering"
        os.makedirs(cache_dir, exist_ok=True)

        filename = os.path.join(cache_dir, 'order_0_' + basename)
        np.save(filename, order_0)

        filename = os.path.join(cache_dir, 'orders_1_and_2' + basename)
        np.save(filename, orders_1_and_2)

    scattering_coef = np.concatenate([order_0, orders_1_and_2], axis=1)
    return scattering_coef

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='energy_scattering',
                    description='Prediction of molecule energy from \
                        configuration using 3D wavelet scattering.')
    parser.add_argument("--data", "-d", help="base folder where data is stored", default="data", dest="data")
    group_scat = parser.add_argument_group("scattering options")
    group_scat.add_argument("--batch_size", help="Batch size for scattering", default=16)
    group_scat.add_argument("-j", help="number of scales", default=2, dest="J")
    group_scat.add_argument("-l", help="number of l values", default=3, dest="L")
    group_reg = parser.add_argument_group("regression options")
    group_reg.add_argument("--alpha", help="alpha for ridge regression", default=1e-5)
    group_reg.add_argument("--val_size", help="validation size in percentage", default=0.05)
    args = parser.parse_args()
    try: scattering_coef_train, scattering_coef_test = load_scattering(folder=args.data)
    except FileNotFoundError: 
        print("Scattering files not found, expected format in data folder: order_0_train_qm7.npy, \
orders_1_and_2_train_qm7.npy, order_0_test_qm7.npy, orders_1_and_2_test_qm7.npy")
        print("Computing scattering from RAW data, will take ~3h with GPU P100.")
        pos_train, charges_train, valence_charges_train, energies_train = load_dataset(args.data)
        pos_train = normalize_pos(pos_train, charges_train)
        scattering_coef_train = scattering(pos_train, charges_train, valence_charges_train, "train",
                                           batch_size=args.batch_size, J=args.J, L=args.L)
        pos_test, charges_test, valence_charges_test = load_dataset(args.data, test=True)
        pos_test = normalize_pos(pos_test, charges_test)
        scattering_coef_test = scattering(pos_test, charges_test, valence_charges_test, "test",
                                          batch_size=args.batch_size, J=args.J, L=args.L)

    full_path_energies = os.path.join(args.data, "energies", "train.csv")
    with open(full_path_energies, 'r') as energies:
        csv_lines = energies.readlines()[1:]
        energies_train = np.zeros((len(csv_lines)), dtype=np.float32)
        for i, csv_line in enumerate(csv_lines):
            _, energy = csv_line.split(',')
            energies_train[i] = float(energy)
    # Split train and validation set
    x_train, x_val, y_train, y_val = model_selection.train_test_split(scattering_coef_train, energies_train, test_size=args.val_size)

    # Set up Ridge regression
    scaler = preprocessing.StandardScaler()
    ridge = linear_model.Ridge(alpha=args.alpha)

    regressor = pipeline.make_pipeline(scaler, ridge)
    regressor.fit(X=x_train, y=y_train)
    val_prediction = regressor.predict(X=x_val)
    RMSE_val = np.sqrt(np.mean((val_prediction - y_val) ** 2))
    train_prediction = regressor.predict(X=x_train)
    RMSE_train = np.sqrt(np.mean((train_prediction - y_train) ** 2))
    print("RMSE training = ", RMSE_train)
    print("RMSE validation = ", RMSE_val)

    energy_test = regressor.predict(X=scattering_coef_test)
    id_test = range(3793, 4740)
    with open("test.csv", 'w') as file:
        file.write("id,energy\n")
        for i, e in zip(id_test, energy_test):
            file.write(f"{i},{e}\n")