"""
Linear regression on scattering coefficients.
"""
import numpy as np
from keras.losses import mean_squared_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from data.loader import load
from tools.featuring.scattering import load_scattering_coefficients
from tools.io import to_kaggle_csv

tqdm.pandas()

# retrieve of test data
train = load(molecules_folder_path=r"../data/atoms/train",
             energies_file_path=r"../data/energies/train.csv",
             already_saved_file_path=r"../data/train.csv")

# retrieve test data
test = load(molecules_folder_path=r"../data/atoms/test",
            energies_file_path=r"../data/energies/test.csv",
            already_saved_file_path=r"../data/test.csv")

# recuperation of train data scattering coefficients
train_scattering = load_scattering_coefficients(train['molecule_id'].drop_duplicates(),
                                                folder_path=r"../data/scattering",
                                                already_saved_file_path="../data/train_scattering.csv")

# recuperation of test data scattering coefficients
test_scattering = load_scattering_coefficients(test['molecule_id'].drop_duplicates(),
                                               folder_path=r"../data/scattering",
                                               already_saved_file_path=r"../data/test_scattering.csv")

# add of molecule energy
train_scattering = train_scattering.merge(train[['molecule_id', 'molecule_energy']].drop_duplicates(),
                                          on='molecule_id', how='left')

# definition and train of linear regression model
model = LinearRegression()
model.fit(train_scattering.drop(columns=['molecule_id', 'molecule_energy']).values,
          train_scattering['molecule_energy'].to_numpy().ravel())

# prediction over test and train data
train_scattering['molecule_energy_prediction'] = model.predict(
    train_scattering.drop(columns=['molecule_id', 'molecule_energy']).values)
test_scattering['molecule_energy'] = model.predict(test_scattering.drop(columns=['molecule_id']).values)

print('RMSE on train : ',
      np.sqrt(mean_squared_error(train_scattering['molecule_energy'], train_scattering['molecule_energy_prediction'])))

to_kaggle_csv(test_scattering, csv_file_path=r"../data/kaggle/scattering_regression.csv",
              molecule_id_column='molecule_id',
              molecule_prediction_column='molecule_energy')
