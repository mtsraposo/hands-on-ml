import numpy as np
import pandas as pd


def create_categories(data):
    data['income_cat'] = pd.cut(data['median_income'],
                                bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    return data


def drop_categories(train_test):
    for label, set_ in train_test.items():
        train_test[label] = set_.drop(['id', 'income_cat'], axis=1)
    return train_test


def append_ratios(data):
    data['rooms_per_household'] = data['total_rooms']/data['households']
    data['bedrooms_per_room'] = data['total_bedrooms']/data['total_rooms']
    data['population_per_household'] = data['population']/data['households']
    return data