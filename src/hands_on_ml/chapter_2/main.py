import os, re

from src.hands_on_ml.chapter_2 import input_data, test_set, \
    feature_engineering, config, visualize, lib_io


def set_directories(config_data):
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = ''.join(re.split('(hands-on-ml)', cur_file_path)[:2])
    work_dir = os.path.join(root_dir, config_data['base_path'])
    os.chdir(work_dir)
    lib_io.create_directories(['resources',
                               config_data['output_path'],
                               config_data['data_path']])


if __name__ == "__main__":
    set_directories(config_data=config.CONFIG_DATA)
    housing_path = input_data.fetch_housing_data(config.CONFIG_DATA)
    housing_raw = input_data.load_housing_data(housing_path)
    visualize.input_data(housing_raw, config.CONFIG_DATA)

    housing_with_id = test_set.add_unique_id(housing_raw)
    housing = feature_engineering.create_categories(housing_with_id)
    train_test = test_set.split_train_test_stratified(housing)
    train_test = feature_engineering.drop_categories(train_test)
    stat_train_set = train_test['train'].copy()

    visualize.training_set(stat_train_set, config.CONFIG_DATA)
