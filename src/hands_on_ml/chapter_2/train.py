from sklearn.linear_model import LinearRegression


def run(housing_prepared, housing_labels):
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    return lin_reg
