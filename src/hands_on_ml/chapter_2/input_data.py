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
    housing_tgz.extractall(path=config_data['housing_path'])
    housing_tgz.close()


def load_housing_data(config_data):
    csv_path = os.path.join(config_data['housing_path'], "housing.csv")
    return pd.read_csv(csv_path)


def inspect_data(df):
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
    os.makedirs('src/hands_on_ml/chapter_2/resources', exist_ok=True)
    fig.savefig('src/hands_on_ml/chapter_2/resources/housing_data.png')


def inspect_training_set(df):
    plt.show()
