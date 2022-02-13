import os
import tarfile

import matplotlib.pyplot as plt
import pandas as pd
from six.moves import urllib


def fetch_housing_data(config_data):
    if not os.path.isdir(config_data['housing_path']):
        os.makedirs(config_data['housing_path'])
    tgz_path = os.path.join(config_data['housing_path'], "housing.tgz")
    housing_url = os.path.join(config_data['download_root'], config_data['download_path'])
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_path = os.path.join(config_data['resources_path'], config_data['housing_path'])
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    return os.path.join(housing_path, "housing.csv")


def load_housing_data(housing_path):
    return pd.read_csv(housing_path)


def inspect_data(df, config_data):
    # Useful DataFrame descriptors
    print("<<< INFO >>>>\n")
    df.info()
    print(f"\n\n"
          f"<<< VALUE COUNTS >>>\n"
          f"{df['ocean_proximity'].value_counts()}\n\n"
          f"<<< DESCRIPTION >>>\n"
          f"{df.describe()}\n\n")

    # Draw a histogram for each numerical attribute
    fig, ax = plt.subplots(figsize=(12, 8))
    df.hist(bins=50, ax=ax)
    os.makedirs(config_data['resources_path'], exist_ok=True)
    fig.savefig(os.path.join(config_data['resources_path'], 'housing_data.png'))


def inspect_training_set(df, config_data):
    fig, ax = plt.subplots()
    df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
            s=df['population'] / 100, label='population', figsize=(10, 7),
            c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
            ax=ax
            )
    plt.legend()
    os.makedirs(config_data['resources_path'], exist_ok=True)
    fig.savefig(os.path.join(config_data['resources_path'], 'training_set.png'))
