import os

import pandas as pd
from matplotlib import pyplot as plt

from pandas.plotting import scatter_matrix

import io


def input_data(df, config_data):
    """
    Useful DataFrame descriptors
    """
    # Textual descriptions
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()

    pd.set_option('display.max_columns', 50)
    info_file = open(os.path.join(config_data['output_path'], 'input_data_info.txt'), 'w')
    info_file.write(f"<<< INFO >>>>\n"
                    f"{df_info}\n\n"
                    f"<<< VALUE COUNTS >>>\n"
                    f"{df['ocean_proximity'].value_counts()}\n\n"
                    f"<<< DESCRIPTION >>>\n"
                    f"{df.describe()}\n\n")
    info_file.close()

    # Draw a histogram for each numerical attribute
    df.hist(bins=50, figsize=(12, 8))
    plt.savefig(os.path.join(config_data['output_path'], 'housing_data.png'))


def plot_price_scatter(df, config_data):
    df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
            s=df['population'] / 100, label='population', figsize=(10, 7),
            c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
            )
    plt.legend()
    plt.savefig(os.path.join(config_data['output_path'], 'training_set.png'))


def training_set(df, config_data):
    plot_price_scatter(df, config_data)

    df.corr()['median_house_value'].sort_values(ascending=False)

    attributes = ['median_house_value', 'median_income', 'total_rooms',
                  'housing_median_age']
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.savefig(os.path.join(config_data['output_path'], 'correlations.png'))