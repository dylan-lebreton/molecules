# -*- coding: utf-8 -*-

"""
gcn.py

File predicting molecule energy using graph convolutional network.
"""
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from scipy.spatial.distance import squareform, pdist
from spektral.data import Dataset, BatchLoader, Graph
from tqdm import tqdm
from spektral.utils import normalized_adjacency
from spektral.layers import GATConv


from data.loader import load_train_val_test
from tools.featuring import compute_atomic_number, compute_valence_number

tqdm.pandas()


class Molecules(Dataset):
    """
    Class defining a dataset of graphs, one graph per molecule provided in molecules_dataframe.
    """

    def __init__(self, molecules_dataframe: pd.DataFrame, **kwargs):
        self.dataframe = molecules_dataframe
        super().__init__(**kwargs)

    def read(self):
        # creation of a list of graphs... one graph per molecule
        result = list()
        for molecule_id in self.dataframe.molecule_id.unique():
            molecule_dataframe = self.dataframe.loc[self.dataframe['molecule_id'] == molecule_id].copy(deep=True)
            molecule_dataframe = molecule_dataframe.reset_index(drop=True)
            result.append(self.generate_graph(molecule_dataframe))
        return result
        # return [Graph(n_nodes=np.random.randint(10, 20), n_node_features=2) for _ in range(self.n_graphs)]

    @staticmethod
    def generate_graph(molecule_dataframe: pd.DataFrame):
        # Generate node features, assuming atomic_number, valence_number, and n_atoms are node features
        node_features = molecule_dataframe[['atomic_number', 'valence_number']].values.astype("float32")

        # Generate adjacency matrix based on Euclidean distance
        coordinates = molecule_dataframe[['x', 'y', 'z']].values
        adjacency_matrix = squareform(pdist(coordinates))

        # Set a threshold distance to consider two atoms as connected (for simplicity, based on your domain knowledge)
        threshold = 1.0
        adjacency_matrix[adjacency_matrix > threshold] = 0
        adjacency_matrix = normalized_adjacency(adjacency_matrix)

        # Generate target value (molecule energy)
        y = molecule_dataframe['molecule_energy'].values[0]

        return Graph(x=node_features, a=adjacency_matrix, y=y)


if __name__ == "__main__":

    train, val, test = load_train_val_test(molecules_folder_path=r"../data/atoms/train",
                                           energies_file_path=r"../data/energies/train.csv",
                                           already_saved_file_path=r"../data/train.csv",
                                           train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42)

    # compute some features
    for data in [train, val, test]:
        # atomic number
        data['atomic_number'] = data['atom_name'].progress_apply(lambda name: compute_atomic_number(name))

        # valence number
        data['valence_number'] = data['atomic_number'].progress_apply(
            lambda atomic_number: compute_valence_number(atomic_number))

    # creation of data loaders
    train_loader = BatchLoader(Molecules(train), batch_size=32)
    val_loader = BatchLoader(Molecules(val), batch_size=32)
    test_loader = BatchLoader(Molecules(test), batch_size=32)

    # model definition
    # Define the inputs
    X_in = Input(shape=(None, 2))  # 2 features for each node
    A_in = Input((None,), sparse=True)  # adjacency matrix

    # Define the model
    X = GATConv(32, activation='relu')([X_in, A_in])
    X = GATConv(1, activation='linear')([X, A_in])

    # Build the model
    model = Model(inputs=[X_in, A_in], outputs=X)

    # optimizer and loss
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    train_loss_history = []
    val_loss_history = []

    epochs = 20

    # Train the model
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Training loop
        epoch_train_loss = 0
        for batch in train_loader:
            inputs, target = batch
            with tf.GradientTape() as tape:
                predictions = model([inputs[0], inputs[1]], training=True)
                loss = loss_fn(target, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_train_loss += loss.numpy()
        epoch_train_loss /= len(train_loader)  # Average training loss for the epoch
        train_loss_history.append(epoch_train_loss)

        # Validation loop
        epoch_val_loss = 0
        for batch in val_loader:
            inputs, target = batch
            predictions = model([inputs[0], inputs[1]], training=False)
            loss = loss_fn(target, predictions)
            epoch_val_loss += loss.numpy()
        epoch_val_loss /= len(val_loader)  # Average validation loss for the epoch
        val_loss_history.append(epoch_val_loss)

        print(f'Epoch {epoch}, Training loss: {epoch_train_loss}, Validation loss: {epoch_val_loss}')

    print("oui")