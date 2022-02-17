import logging

import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from src.hands_on_ml.chapter_2 import feature_engineering


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


def show_importances(grid_search, pipeline, num_attribs, extra_attribs):
    attributes = feature_engineering.get_features_names(pipeline, num_attribs, extra_attribs)
    feature_importances = grid_search.best_estimator_.feature_importances_
    importance_list = sorted(zip(feature_importances, attributes), reverse=True)
    [logging.info(i) for i in importance_list]
    return importance_list


def search_hyperparameters(housing_model, config_evaluation):
    regressor_type = getattr(sklearn, config_evaluation['regressor']['type'])
    model = getattr(regressor_type, config_evaluation['regressor']['name'])
    # An alternative would be RandomizedSearchCV, with a limit
    # on the number of iterations
    grid_search = GridSearchCV(model(), config_evaluation['param_grid'], cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_model['prepared_data'], housing_model['training']['labels'])
    return grid_search


def run(housing_model, config_evaluation, config_preproc):
    model_evaluation = ModelEvaluation(housing_model['prepared_data'],
                                       housing_model['model'],
                                       housing_model['training']['labels'])
    eval_function = getattr(ModelEvaluation, config_evaluation['method'])
    metrics = eval_function(model_evaluation)

    hyperparameter_search = search_hyperparameters(housing_model, config_evaluation)
    logging.info(hyperparameter_search.best_params_)
    show_importances(hyperparameter_search,
                     housing_model['pipeline'],
                     housing_model['attributes']['num_attribs'], config_preproc['extra_features'])
    return {'metrics': metrics,
            'hyperparameter_search': hyperparameter_search}
