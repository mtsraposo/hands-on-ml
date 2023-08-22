from scipy.stats import expon, reciprocal

CONFIG_DATA = {'download_root': 'https://raw.githubusercontent.com/ageron/handson-ml2/master/',
               'download_path': 'datasets/housing/housing.tgz',
               'base_path': 'src/hands_on_ml/chapter_2',
               'output_path': 'resources/output',
               'data_path': 'resources/data'}

CONFIG_PREPROC = {'categorial_attributes': ['ocean_proximity'],
                  'numerical_attributes_to_ratios': ['total_rooms', 'households',
                                                     'total_bedrooms', 'population'],
                  'extra_features': ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']}

CONFIG_TRAIN = {'algorithm': {'type': 'svm',
                              'name': 'SVR',
                              'params': {'kernel': 'linear',
                                         'C': 30000}},
                'grid_search': {'name': 'RandomizedSearchCV',
                                'params': {
                                    'param_distributions': [
                                        {'model__kernel': ['linear', 'rbf'],
                                         'model__C': reciprocal(20, 200000),
                                         'model__gamma': expon(scale=1.0)}
                                    ],
                                    'cv': 5,
                                    'scoring': 'neg_mean_squared_error',
                                    'return_train_score': True,
                                    'n_iter': 6,
                                    'verbose': 2
                                }}
                }

CONFIG_EVALUATION = {'method': 'cross_validation',
                     'confidence': 0.95}

CONFIG_OUTPUT = {'path': 'resources/output/housing.joblib'}
