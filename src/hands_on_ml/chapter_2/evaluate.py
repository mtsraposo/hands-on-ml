import logging

import numpy as np
import sklearn
from scipy import stats
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

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
        scores = model_selection.cross_val_score(self.model, self.prepared_data, self.labels,
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
    search_class = getattr(model_selection, config_evaluation['search_class']['name'])
    grid_search = search_class(model(), **config_evaluation['search_class']['params'])
    grid_search.fit(housing_model['prepared_data'], housing_model['split']['training_labels'])
    return grid_search


def predict_test_set(hyperparameters, strat_test_set, full_pipeline):
    final_model = hyperparameters.best_estimator_
    x_test = strat_test_set.drop('median_house_value', axis=1)
    x_test_prepared = full_pipeline.transform(x_test)
    return final_model.predict(x_test_prepared)


def calc_out_sample_rmse(strat_test_set, test_set_predictions):
    mse = mean_squared_error(strat_test_set['median_house_value'], test_set_predictions)
    rmse = np.sqrt(mse)
    logging.info(f'RMSE: {rmse}')
    return rmse


def gen_rmse_confidence_interval(final_predictions, strat_test_set, confidence):
    squared_errors = (final_predictions - strat_test_set['median_house_value']) ** 2
    confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                                   loc=squared_errors.mean(),
                                                   scale=stats.sem(squared_errors)))
    logging.info(f'RMSE ({100 * confidence: .0%}): {confidence_interval}')
    return confidence_interval


def eval_test_set(hyperparameters, strat_test_set, full_pipeline, confidence):
    test_set_predictions = predict_test_set(hyperparameters, strat_test_set, full_pipeline)
    return {'predictions': test_set_predictions,
            'rmse': calc_out_sample_rmse(strat_test_set, test_set_predictions),
            'confidence_interval': gen_rmse_confidence_interval(test_set_predictions, strat_test_set, confidence)}


def run(housing_model, config_evaluation, config_preproc):
    model_evaluation = ModelEvaluation(housing_model['prepared_data'],
                                       housing_model['model'],
                                       housing_model['split']['training_labels'])
    eval_function = getattr(ModelEvaluation, config_evaluation['method'])
    metrics = eval_function(model_evaluation)

    hyperparameters = search_hyperparameters(housing_model, config_evaluation)
    logging.info(hyperparameters.best_params_)
    show_importances(hyperparameters,
                     housing_model['pipeline'],
                     housing_model['attributes']['num_attribs'], config_preproc['extra_features'])

    test_set_rmse = eval_test_set(hyperparameters,
                                  strat_test_set=housing_model['split']['test'],
                                  full_pipeline=housing_model['pipeline'],
                                  confidence=config_evaluation['confidence'])
    return {'metrics': metrics,
            'hyperparameters': hyperparameters,
            'test': test_set_rmse}
