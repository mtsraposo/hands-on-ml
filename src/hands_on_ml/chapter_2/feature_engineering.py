from functools import reduce

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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


def index_attributes(df, attributes):
    return {col: list(df.columns).index(col) for col in attributes}


def add_unique_id(data):
    data['id'] = data['longitude'] * 1000 + data['latitude']
    return data


def create_categories(data):
    data['income_cat'] = pd.cut(data['median_income'],
                                bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    return data


def get_features_names(pipeline, num_attribs, extra_attribs):
    """
    Returns the list of feature names, as defined by the pipeline
    This implementation explicitly places each attribute type according
    to the order in which it appears in the pipeline, so it must be updated
    when the pipeline changes
    """
    cat_encoder = pipeline.named_transformers_['cat']
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    return num_attribs + extra_attribs + cat_one_hot_attribs


def run(housing_raw):
    return reduce(lambda res, f: f(res),
                  [add_unique_id,
                   create_categories],
                  housing_raw)
