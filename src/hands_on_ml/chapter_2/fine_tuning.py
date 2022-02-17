from sklearn.model_selection import GridSearchCV
import sklearn
import logging


def run(prepared_data, labels, param_grid, regressor):
    regressor_type = getattr(sklearn, regressor['type'])
    model = getattr(regressor_type, regressor['name'])
    grid_search = GridSearchCV(model(), param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(prepared_data, labels)
    logging.info(grid_search.best_params_)
    return grid_search
