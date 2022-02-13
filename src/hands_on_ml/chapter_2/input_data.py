import os
import tarfile

import pandas as pd
from six.moves import urllib


def fetch_housing_data(config_data):
    housing_path = os.path.join(config_data['housing_path'], 'housing')
    tgz_path = os.path.join(housing_path, "housing.tgz")
    housing_url = os.path.join(config_data['download_root'], config_data['download_path'])

    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

    return os.path.join(housing_path, "housing.csv")


def load_housing_data(housing_path):
    return pd.read_csv(housing_path)
