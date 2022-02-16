from sklearn.metrics import mean_squared_error
import numpy as np


def predict_sample(data, labels, pipeline, model):
    sample_prepared = pipeline.transform(data)
    print('Predictions: ', model.predict(sample_prepared))
    print('Labels: ', list(labels))


def rmse(data_prepared, model, labels):
    housing_predictions = model.predict(data_prepared)
    lin_mse = mean_squared_error(labels,
                                 housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    return lin_rmse


def run(data_prepared, model, labels):
    lin_rmse = rmse(data_prepared, model, labels)
    print(f'RMSE: {lin_rmse:.2f}')
