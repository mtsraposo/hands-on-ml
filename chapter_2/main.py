from chapter_2 import input_data, test_set, feature_engineering

if __name__ == "__main__":
    input_data.fetch_housing_data()
    housing_raw = input_data.load_housing_data()
    input_data.inspect_data(housing_raw)

    housing_with_id = test_set.add_unique_id(housing_raw)
    housing = feature_engineering.create_categories(housing_with_id)
    train_test = test_set.split_train_test_stratified(housing)
    train_test = feature_engineering.drop_categories(train_test)

