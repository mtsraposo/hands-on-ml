import logging

from src.hands_on_ml.chapter_2 import input_data, split, feature_engineering, \
    config, preprocess, train, evaluate, \
    lib_io

logging.basicConfig(level=logging.INFO, format='%(message)s')


def run(config_data, config_preproc, config_train):
    housing_raw = input_data.run(config_data)
    housing_raw = feature_engineering.run(housing_raw)
    sample_split = split.run(housing_raw, config_data)
    attributes = preprocess.get_attributes(housing=sample_split['training_set'],
                                           config_preproc=config_preproc)
    housing_pipeline = preprocess.gen_full_pipeline(attributes, config_preproc)
    housing_prepared = preprocess.run(training_set=sample_split['training_set'],
                                      labels=sample_split['training_labels'],
                                      full_pipeline=housing_pipeline)
    return {'split': sample_split,
            'attributes': attributes,
            'prepared_data': housing_prepared,
            'pipeline': housing_pipeline,
            'model': train.run(housing_prepared,
                               housing_labels=sample_split['training_labels'],
                               training_algo=config_train['algorithm'])}


if __name__ == "__main__":
    housing_model = run(config_data=config.CONFIG_DATA,
                        config_preproc=config.CONFIG_PREPROC,
                        config_train=config.CONFIG_TRAIN)
    lib_io.persist_model(housing_model['model'], config.CONFIG_OUTPUT['path'])
    model_evaluation = evaluate.run(housing_model,
                                    config_evaluation=config.CONFIG_EVALUATION,
                                    config_preproc=config.CONFIG_PREPROC)
