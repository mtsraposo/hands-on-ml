from sklearn.model_selection import GridSearchCV
import sklearn
import logging
from src.hands_on_ml.chapter_2 import feature_engineering


def show_importances(grid_search, pipeline, num_attribs, extra_attribs):
    attributes = feature_engineering.get_features_names(pipeline, num_attribs, extra_attribs)
    feature_importances = grid_search.best_estimator_.feature_importances_
    importance_list = sorted(zip(feature_importances, attributes), reverse=True)
    logging.info(importance_list)
    return importance_list


def run(housing_model, config_tuning, config_preproc):
    regressor_type = getattr(sklearn, config_tuning['regressor']['type'])
    model = getattr(regressor_type, config_tuning['regressor']['name'])
    # An alternative would be RandomizedSearchCV, with a limit
    # on the number of iterations
    grid_search = GridSearchCV(model(), config_tuning['param_grid'], cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_model['prepared_data'], housing_model['training']['labels'])
    logging.info(grid_search.best_params_)
    show_importances(grid_search,
                     housing_model['pipeline'],
                     housing_model['attributes']['num_attribs'], config_preproc['extra_features'])
    return grid_search
