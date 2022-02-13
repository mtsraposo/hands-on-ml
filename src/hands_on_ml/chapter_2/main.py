from src.hands_on_ml.chapter_2 import input_data, test_set, \
    feature_engineering, config

if __name__ == "__main__":
    housing_path = input_data.fetch_housing_data(config.CONFIG_DATA)
    housing_raw = input_data.load_housing_data(housing_path)
    input_data.inspect_data(housing_raw, config.CONFIG_DATA)

    housing_with_id = test_set.add_unique_id(housing_raw)
    housing = feature_engineering.create_categories(housing_with_id)
    train_test = test_set.split_train_test_stratified(housing)
    train_test = feature_engineering.drop_categories(train_test)
    stat_train_set = train_test['train'].copy()

    input_data.inspect_training_set(stat_train_set, config.CONFIG_DATA)
