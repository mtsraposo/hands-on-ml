from chapter_2 import input_data, test_set

if __name__ == "__main__":
    input_data.fetch_housing_data()
    housing = input_data.load_housing_data()
    input_data.inspect_data(housing)

    housing = test_set.add_unique_id(housing)
    train_test = test_set.split_train_test_by_id(housing, 0.2, 'id')

