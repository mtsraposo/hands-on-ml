from src.hands_on_ml.chapter_2 import input_data, split, feature_engineering, config, preprocessing, train, evaluate, \
    lib_io

if __name__ == "__main__":
    housing_raw = input_data.run(config_data=config.CONFIG_DATA)
    housing_with_features = feature_engineering.run(housing_raw)
    housing_training = split.run(housing_raw, config.CONFIG_DATA)
    housing_pipeline = preprocessing.gen_full_pipeline(housing=housing_training['set'],
                                                       config_preproc=config.CONFIG_PREPROC)
    housing_prepared = preprocessing.run(housing=housing_training['set'],
                                         full_pipeline=housing_pipeline)
    model = train.run(housing_prepared,
                      housing_labels=housing_training['labels'],
                      method=config.CONFIG_TRAIN['method'])

    housing_evaluation = evaluate.run(prepared_data=housing_prepared,
                                      model=model,
                                      labels=housing_training['labels'],
                                      method=config.CONFIG_EVALUATION['method'])

    lib_io.persist_model(model, config.CONFIG_OUTPUT['path'])
