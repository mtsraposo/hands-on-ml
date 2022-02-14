import pandas as pd
from sklearn.impute import SimpleImputer

def fill_with_median(housing):
    imputer = SimpleImputer(strategy='median')
    housing_num = housing.drop('ocean_proximity', axis=1) # drop non-numerical column
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    return pd.DataFrame(X, columns=housing_num.columns)

