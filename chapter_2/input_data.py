import os
import tarfile

import matplotlib.pyplot as plt
import pandas as pd
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
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
    os.makedirs('chapter_2/resources', exist_ok=True)
    fig.savefig('chapter_2/resources/housing_data.png')
