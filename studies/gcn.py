# -*- coding: utf-8 -*-

"""
gcn.py

File predicting molecule energy using graph convolutional network.
"""
from keras import Model
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from spektral.data import Dataset, BatchLoader
from spektral.layers import GCNConv
from spektral.layers import GlobalSumPool, GraphMasking
from tqdm import tqdm

from data.loader import load, load_train_test
from tools.featuring.basic import compute_atomic_number, compute_valence_number
from tools.featuring.gcn import generate_graph

tqdm.pandas()


class Net(Model):
    def __init__(self):
        super().__init__()
        self.masking = GraphMasking()
        self.conv1 = GCNConv(32, activation="relu")
        self.conv2 = GCNConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense = Dense(1)

    def call(self, inputs):
        x, a = inputs
        x = self.masking(x)
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.global_pool(x)
        output = self.dense(output)

        return output


class MoleculeDataset(Dataset):
    """
    Definition of a class that will be used to store the graph representation of the molecules.
    """

    def __init__(self, **kwargs):
        self.graphs = []
        super().__init__(**kwargs)

    def read(self):
        # This method is supposed to return a list of Graphs
        # Since we already have the list, we simply return it
        return self.graphs

    def add_graph(self, graph):
        self.graphs.append(graph)


if __name__ == '__main__':

    learning_rate = 1e-2  # Learning rate
    epochs = 20  # Number of training epochs
    batch_size = 32  # Batch size

    # retrieve of train and validation data
    train, validation = load_train_test(molecules_folder_path = r"../data/atoms/train", 
                                        energies_file_path=r"../data/energies/train.csv",
                                        already_saved_file_path=r"../data/train.csv",
                                        train_ratio = 0.8, test_ratio = 0.2,
                                        random_state = 42) 

    #########################################################
    # Feature engineering and dataset creation for train data
    #########################################################

    # computation of atomic number
    tqdm.pandas(desc="Computing Atomic Number")
    train['Z'] = train['atom_name'].progress_apply(lambda name: compute_atomic_number(name))

    # sort the dataframe by molecule_id and atome atomic number
    train.sort_values(by=['molecule_id', 'Z'], inplace=True)

    # computation of valence number
    tqdm.pandas(desc="Computing Valence Number")
    train['valence_number'] = train['Z'].progress_apply(lambda atomic_number: compute_valence_number(atomic_number))

    # creation of train dataset
    train_dataset = MoleculeDataset()
    for molecule_id in tqdm(train.molecule_id.unique(), desc="Creating train dataset"):
        molecule_df = train[train.molecule_id == molecule_id].reset_index(drop=True)
        molecule_graph = generate_graph(molecule_df, "x", "y", "z", charges_column="Z",
                                        valence_column="valence_number", molecule_energy_column="molecule_energy")
        train_dataset.add_graph(molecule_graph)
        
    ##############################################################
    # Feature engineering and dataset creation for validation data
    ##############################################################

    # computation of atomic number
    tqdm.pandas(desc="Computing Atomic Number")
    validation['Z'] = validation['atom_name'].progress_apply(lambda name: compute_atomic_number(name))

    # sort the dataframe by molecule_id and atome atomic number
    validation.sort_values(by=['molecule_id', 'Z'], inplace=True)

    # computation of valence number
    tqdm.pandas(desc="Computing Valence Number")
    validation['valence_number'] = validation['Z'].progress_apply(lambda atomic_number: compute_valence_number(atomic_number))

    # creation of validation dataset
    validation_dataset = MoleculeDataset()
    for molecule_id in tqdm(validation.molecule_id.unique(), desc="Creating validation dataset"):
        molecule_df = validation[validation.molecule_id == molecule_id].reset_index(drop=True)
        molecule_graph = generate_graph(molecule_df, "x", "y", "z", charges_column="Z",
                                        valence_column="valence_number", molecule_energy_column="molecule_energy")
        validation_dataset.add_graph(molecule_graph)

    ########################################################
    # Feature engineering and dataset creation for test data
    ########################################################

    # retrieve test data
    test = load(molecules_folder_path=r"../data/atoms/test",
                energies_file_path=r"../data/energies/test.csv",
                already_saved_file_path=r"../data/test.csv")

    # computation of atomic number
    tqdm.pandas(desc="Computing Atomic Number")
    test['Z'] = test['atom_name'].progress_apply(lambda name: compute_atomic_number(name))

    # sort the dataframe by molecule_id and atome atomic number
    test.sort_values(by=['molecule_id', 'Z'], inplace=True)

    # computation of valence number
    tqdm.pandas(desc="Computing Valence Number")
    test['valence_number'] = test['Z'].progress_apply(lambda atomic_number: compute_valence_number(atomic_number))

    # creation of test dataset
    test_dataset = MoleculeDataset()
    for molecule_id in tqdm(test.molecule_id.unique(), desc="Creating test dataset"):
        molecule_df = test[test.molecule_id == molecule_id].reset_index(drop=True)
        molecule_graph = generate_graph(molecule_df, "x", "y", "z", charges_column="Z",
                                        valence_column="valence_number", molecule_energy_column=None)
        test_dataset.add_graph(molecule_graph)

    ############
    # Prediction
    ############

    # create datasets loaders
    train_loader = BatchLoader(train_dataset, batch_size=batch_size, mask=True)
    validation_loader = BatchLoader(validation_dataset, batch_size=batch_size, mask=True)
    test_loader = BatchLoader(test_dataset, batch_size=batch_size, mask=True)

    # build model
    model = Net()
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    # fit the model
    history = model.fit(train_loader.load(),
                        steps_per_epoch=train_loader.steps_per_epoch, epochs=epochs,
                        validation_data=validation_loader.load(), validation_steps=validation_loader.steps_per_epoch)

    # plot training and validation loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evolution of loss on train and validation data')
    plt.legend()
    plt.show()

    # Predict model
    predictions = model.predict(test_loader.load(), steps=test_loader.steps_per_epoch)
