import logging

import numpy as np
from scipy import stats
from sklearn import model_selection
from sklearn.metrics import mean_squared_error


def predict_sample(data, labels, pipeline, model):
    sample_prepared = pipeline.transform(data)
    print('Predictions: ', model.predict(sample_prepared))
    print('Labels: ', list(labels))


class ModelEvaluation:
    def __init__(self, data_to_eval, model, labels):
        self.data_to_eval = data_to_eval
        self.model = model
        self.labels = labels

    def rmse(self):
        housing_predictions = self.model.predict(self.data_to_eval)
        eval_mse = mean_squared_error(self.labels,
                                      housing_predictions)
        eval_rmse = np.sqrt(eval_mse)
        print(f'RMSE: {eval_rmse}')
        return eval_rmse

    def cross_validation(self):
        scores = model_selection.cross_val_score(self.model, self.data_to_eval, self.labels,
                                                 scoring='neg_mean_squared_error', cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        print(f'Scores: {tree_rmse_scores}')
        print(f'Mean: {tree_rmse_scores.mean():.2f}')
        print(f'Standard deviation: {tree_rmse_scores.std():.2f}')
        return tree_rmse_scores


def get_features_names(trained, extra_features):
    """
    Returns the list of feature names, as defined by the pipeline
    This implementation explicitly places each attribute type according
    to the order in which it appears in the pipeline, so it must be updated
    when the pipeline changes
    """
    cat_encoder = trained['estimators']['preproc'].named_transformers_['cat']
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    return (trained['attributes']['num_attribs']
            + extra_features
            + cat_one_hot_attribs)


def show_importances(trained, extra_features):
    """
    For models such as RandomForestRegressor, this function will show the relative importance of
    each regressed feature.
    """
    attributes = get_features_names(trained, extra_features)
    feature_importances = trained['estimators']['search'].best_estimator_.feature_importances_
    importance_list = sorted(zip(feature_importances, attributes), reverse=True)
    [logging.info(i) for i in importance_list]
    return importance_list


def prepare_test_set(trained):
    x_test = trained['split']['test'].drop('median_house_value', axis=1)
    return trained['estimators']['preproc'].transform(x_test)


def predict_test_set(trained):
    final_model = trained['estimators']['search'].best_estimator_['model']
    x_test_prepared = prepare_test_set(trained)
    return final_model.predict(x_test_prepared)


def calc_out_sample_rmse(trained, test_set_predictions):
    mse = mean_squared_error(trained['split']['test']['median_house_value'], test_set_predictions)
    rmse = np.sqrt(mse)
    logging.info(f'RMSE: {rmse:.2f}')
    return rmse


def gen_rmse_confidence_interval(trained, test_set_predictions, confidence):
    squared_errors = (test_set_predictions - trained['split']['test']['median_house_value']) ** 2
    interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                        loc=squared_errors.mean(),
                                        scale=stats.sem(squared_errors)))
    logging.info(f'RMSE ({confidence:.0%}): [{interval[0]:.2f},{interval[1]:.2f}]')
    return interval


def eval_test_set(trained, confidence):
    test_set_predictions = predict_test_set(trained)
    return {'predictions': test_set_predictions,
            'rmse': calc_out_sample_rmse(trained, test_set_predictions),
            'confidence_interval': gen_rmse_confidence_interval(trained, test_set_predictions, confidence)}


def run(trained, config_evaluation):
    model_evaluation = ModelEvaluation(data_to_eval=trained['estimators']['training_set_prepared'],
                                       model=trained['estimators']['regression'],
                                       labels=trained['split']['training_labels'])
    eval_function = getattr(ModelEvaluation, config_evaluation['method'])
    metrics = eval_function(model_evaluation)

    logging.info(trained['estimators']['search'].best_params_)
    test_set_rmse = eval_test_set(trained,
                                  confidence=config_evaluation['confidence'])
    return {'metrics': metrics,
            'test': test_set_rmse}
