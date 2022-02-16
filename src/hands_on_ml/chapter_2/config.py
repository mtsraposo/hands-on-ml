CONFIG_DATA = {'download_root': 'https://raw.githubusercontent.com/ageron/handson-ml2/master/',
               'download_path': 'datasets/housing/housing.tgz',
               'base_path': 'src/hands_on_ml/chapter_2',
               'output_path': 'resources/output',
               'data_path': 'resources/data'}

CONFIG_PREPROC = {'categorial_attributes': ['ocean_proximity'],
                  'numerical_attributes_to_ratios': ['total_rooms', 'households',
                                                     'total_bedrooms', 'population']}

CONFIG_TRAIN = {'method': 'random_forest'}

CONFIG_EVALUATION = {'method': 'cross_validation'}

CONFIG_OUTPUT = {'path': 'resources/output/housing.joblib'}
