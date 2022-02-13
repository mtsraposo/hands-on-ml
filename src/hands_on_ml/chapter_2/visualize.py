import os

from matplotlib import pyplot as plt

import io


def input_data(df, config_data):
    """
    Useful DataFrame descriptors
    """
    # Textual descriptions
    buf = io.StringIO()
    df.info(buf=buf)
    df_info = buf.getvalue()

    info_file = open(os.path.join(config_data['output_path']),
                     'input_data_info.txt')
    info_file.write(f"<<< INFO >>>>\n"
                    f"{df_info}\n\n"
                    f"<<< VALUE COUNTS >>>\n"
                    f"{df['ocean_proximity'].value_counts()}\n\n"
                    f"<<< DESCRIPTION >>>\n"
                    f"{df.describe()}\n\n")
    info_file.close()

    # Draw a histogram for each numerical attribute
    fig, ax = plt.subplots(figsize=(12, 8))
    df.hist(bins=50, ax=ax)
    fig.savefig(os.path.join(config_data['output_path'],
                             'housing_data.png'))


def training_set(df, config_data):
    fig, ax = plt.subplots()
    df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
            s=df['population'] / 100, label='population', figsize=(10, 7),
            c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
            ax=ax
            )
    plt.legend()
    fig.savefig(os.path.join(config_data['output_path'],
                             'output/training_set.png'))
