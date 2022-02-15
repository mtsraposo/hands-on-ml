import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.hands_on_ml.chapter_2 import feature_engineering


def fill_with_median(housing_num):
    """
    Fills empty values with the median
    This function is left here only for illustration.
    It's not used by the model. Refer to the run() function
    for the pipeline in use.
    """
    imputer = SimpleImputer(strategy='median')
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    return housing_tr


def gen_ordinal_encoding(housing_cat):
    """
    Assigns a numerical value to each category on a given column.
    This function is left here only for illustration.
    It's not used by the model. Refer to the run() function
    for the pipeline in use.
    """
    ordinal_encoder = OrdinalEncoder()
    housing_cat = housing_cat[['ocean_proximity']]
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    return housing_cat_encoded


def gen_onehot_encoding(housing_cat):
    """
    Assigns 1 to 'hot' categories and 0 to other categories on a given column
    This function is left here only for illustration.
    It's not used by the model. Refer to the run() function
    for the pipeline in use.
    """
    cat_encoder = OneHotEncoder()
    housing_cat = housing_cat[['ocean_proximity']]
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)  # returns a SciPy sparse matrix
    return housing_cat_1hot


def gen_numerical_pipeline(attr_to_ix):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', feature_engineering.CombinedAttributesAdder(attr_to_ix)),
        ('std_scaler', StandardScaler())
    ])


def run(housing, config_preproc):
    cat_attribs = config_preproc['categorial_attributes']
    housing_num = housing.drop(cat_attribs, axis=1)  # drop non-numerical column
    num_attribs = list(housing_num)

    attr_to_ix = feature_engineering.index_attributes(housing_num,
                                                      # use only selected attributes to generate ratios
                                                      attributes=config_preproc['numerical_attributes_to_ratios'])

    num_pipeline = gen_numerical_pipeline(attr_to_ix)
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    ])

    return full_pipeline.fit_transform(housing)
