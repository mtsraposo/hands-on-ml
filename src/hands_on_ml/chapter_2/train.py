from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


class HousingModel:
    def __init__(self, prepared_data, labels):
        self.prepared_data = prepared_data
        self.labels = labels

    def train_lin_reg(self):
        lin_reg = LinearRegression()
        lin_reg.fit(self.prepared_data, self.labels)
        return lin_reg

    def train_tree_reg(self):
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(self.prepared_data, self.labels)
        return tree_reg


def run(housing_prepared, housing_labels, method):
    housing = HousingModel(housing_prepared, housing_labels)
    regression_fun = getattr(HousingModel, f"train_{method}")
    model = regression_fun(housing)
    return model
