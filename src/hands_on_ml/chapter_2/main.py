from src.hands_on_ml.chapter_2 import input_data, split, feature_engineering, config, preprocessing, train

if __name__ == "__main__":
    housing_raw = input_data.run(config_data=config.CONFIG_DATA)
    housing_with_features = feature_engineering.run(housing_raw)
    housing_training = split.run(housing_raw, config.CONFIG_DATA)
    housing_pipeline = preprocessing.gen_full_pipeline(housing=housing_training['set'],
                                                       config_preproc=config.CONFIG_PREPROC)
    housing_prepared = preprocessing.run(housing=housing_training['set'],
                                         full_pipeline=housing_pipeline)
    lin_reg = train.run(housing_prepared, housing_training['labels'])

    some_data = housing_training['set'].iloc[:5]
    some_labels = housing_training['labels'].iloc[:5]
    some_data_prepared = housing_pipeline.transform(some_data)
    print('Predictions: ', lin_reg.predict(some_data_prepared))
    print('Labels: ', list(some_labels))
