import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


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
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    return data


def gen_ordinal_encoding(housing):
    """
    Assigns a numerical value to each category on a given column
    """
    ordinal_encoder = OrdinalEncoder()
    housing_cat = housing[['ocean_proximity']]
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    return housing_cat_encoded


def gen_onehot_encoding(housing):
    """
    Assigns 1 to 'hot' categories and 0 to other categories on a given column
    """
    cat_encoder = OneHotEncoder()
    housing_cat = housing[['ocean_proximity']]
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)  # returns a SciPy sparse matrix
    return housing_cat_1hot


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Custom class implementing the basic transformer methods
    """
    def __init__(self, attr_to_ix, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.attr_to_ix = attr_to_ix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.attr_to_ix['total_rooms']] / X[:, self.attr_to_ix['households']]
        population_per_household = X[:, self.attr_to_ix['population']] / X[:, self.attr_to_ix['households']]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.attr_to_ix['total_bedrooms']] / X[:, self.attr_to_ix['total_rooms']]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def gen_extra_attributes(housing, attributes=None):
    if attributes is None:
        attributes = ['total_rooms', 'households',
                      'total_bedrooms', 'population']
    attr_to_ix = {col: list(housing.columns).index(col) for col in attributes}
    attr_adder = CombinedAttributesAdder(attr_to_ix, add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)
    return housing_extra_attribs
