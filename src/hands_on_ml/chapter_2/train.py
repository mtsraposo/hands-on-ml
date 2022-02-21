import sklearn


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


def run(prepared_data, labels, training_algo):
    housing = HousingModel(prepared_data, labels)
    model_type = getattr(sklearn, training_algo['type'])
    regression_class = getattr(model_type, training_algo['name'])
    return housing.fit_model(regression_class(**training_algo['params']))
