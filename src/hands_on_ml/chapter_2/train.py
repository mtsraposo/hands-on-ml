from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class HousingModel:
    """
    Implemented here to allow for calling models by their names
    and eventually implementing custom features to each of them
    """
    def __init__(self, prepared_data, labels):
        self.prepared_data = prepared_data
        self.labels = labels

    def fit_model(self, model):
        model.fit(self.prepared_data, self.labels)
        return model

    def lin_reg(self):
        return self.fit_model(LinearRegression())

    def tree_reg(self):
        return self.fit_model(DecisionTreeRegressor())

    def random_forest(self):
        return self.fit_model(RandomForestRegressor())


def run(housing_prepared, housing_labels, method):
    housing = HousingModel(housing_prepared, housing_labels)
    regression_fun = getattr(HousingModel, method)
    return regression_fun(housing)
