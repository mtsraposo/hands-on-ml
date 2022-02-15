from src.hands_on_ml.chapter_2 import input_data, split, feature_engineering, config, preprocessing

if __name__ == "__main__":
    housing_raw = input_data.run(config_data=config.CONFIG_DATA)
    housing_with_features = feature_engineering.run(housing_raw)
    housing_training = split.run(housing_raw, config.CONFIG_DATA)
    housing = preprocessing.run(housing=housing_training['set'],
                                config_preproc=config.CONFIG_PREPROC)
