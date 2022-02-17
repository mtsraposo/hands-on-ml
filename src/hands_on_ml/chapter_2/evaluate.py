from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score


def predict_sample(data, labels, pipeline, model):
    sample_prepared = pipeline.transform(data)
    print('Predictions: ', model.predict(sample_prepared))
    print('Labels: ', list(labels))


class ModelEvaluation:
    def __init__(self, prepared_data, model, labels):
        self.prepared_data = prepared_data
        self.model = model
        self.labels = labels

    def rmse(self):
        housing_predictions = self.model.predict(self.prepared_data)
        eval_mse = mean_squared_error(self.labels,
                                      housing_predictions)
        eval_rmse = np.sqrt(eval_mse)
        print(f'RMSE: {eval_rmse}')
        return eval_rmse

    def cross_validation(self):
        scores = cross_val_score(self.model, self.prepared_data, self.labels,
                                 scoring='neg_mean_squared_error', cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        print(f'Scores: {tree_rmse_scores}')
        print(f'Mean: {tree_rmse_scores.mean():.2f}')
        print(f'Standard deviation: {tree_rmse_scores.std():.2f}')
        return tree_rmse_scores


def run(housing_model, config_evaluation):
    model_evaluation = ModelEvaluation(housing_model['prepared_data'],
                                       housing_model['model'],
                                       housing_model['training']['labels'])
    eval_function = getattr(ModelEvaluation, config_evaluation['method'])
    eval_result = eval_function(model_evaluation)
    return eval_result
