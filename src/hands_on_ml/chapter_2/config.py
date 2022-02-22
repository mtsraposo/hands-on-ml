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
                                         'C': 0.5,
                                         'gamma': 'auto'}},
                'grid_search': {'name': 'RandomizedSearchCV',
                                'params': {
                                    'param_distributions': [
                                        {'model__kernel': ['linear', 'rbf']},
                                        {'model__C': [0.2, 0.4, 0.6, 0.8, 1.0]},
                                        {'model__gamma': ['auto', 'scale']},
                                        # {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                                        # {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
                                    ],
                                    'cv': 5,
                                    'scoring': 'neg_mean_squared_error',
                                    'return_train_score': True,
                                    'n_iter': 5
                                }}
                }

CONFIG_EVALUATION = {'method': 'cross_validation',
                     'confidence': 0.95}

CONFIG_OUTPUT = {'path': 'resources/output/housing.joblib'}
