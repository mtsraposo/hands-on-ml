import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from hands_on_ml.chapter_2 import feature_engineering


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


def get_attributes(training_set, config_preproc):
    cat_attribs = config_preproc['categorial_attributes']
    num_df = training_set.drop(cat_attribs, axis=1)
    return {'cat_attribs': cat_attribs,
            'num_df': num_df,  # drop non-numerical column
            'num_attribs': list(num_df)}


def gen_pipeline(attributes, config_preproc):
    attr_to_ix = feature_engineering.index_attributes(attributes['num_df'],
                                                      # use only selected attributes to generate ratios
                                                      attributes=config_preproc['numerical_attributes_to_ratios'])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', feature_engineering.CombinedAttributesAdder(attr_to_ix)),
        ('std_scaler', StandardScaler()),
        ('feature_selector', SelectPercentile(f_regression, percentile=5))
    ])

    preproc_pipeline = ColumnTransformer([
        ('num', num_pipeline, attributes['num_attribs']),
        ('cat', OneHotEncoder(), attributes['cat_attribs']),
    ])

    return preproc_pipeline
