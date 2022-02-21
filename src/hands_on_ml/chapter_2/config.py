CONFIG_DATA = {'download_root': 'https://raw.githubusercontent.com/ageron/handson-ml2/master/',
               'download_path': 'datasets/housing/housing.tgz',
               'base_path': 'src/hands_on_ml/chapter_2',
               'output_path': 'resources/output',
               'data_path': 'resources/data'}

CONFIG_PREPROC = {'categorial_attributes': ['ocean_proximity'],
                  'numerical_attributes_to_ratios': ['total_rooms', 'households',
                                                     'total_bedrooms', 'population'],
                  'extra_features': ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']}

CONFIG_TRAIN = {'method': 'random_forest'}

CONFIG_EVALUATION = {'method': 'cross_validation',
                     'regressor': {'type': 'ensemble',
                                   'name': 'RandomForestRegressor'},
                     'param_grid': [
                         {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                         {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
                     ],
                     'confidence': 0.95}

CONFIG_OUTPUT = {'path': 'resources/output/housing.joblib'}
