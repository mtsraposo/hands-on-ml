import logging

from src.hands_on_ml.chapter_2 import input_data, split, feature_engineering, \
    config, preprocess, train, evaluate, \
    lib_io

logging.basicConfig(level=logging.INFO, format='%(message)s')


def run(config_data, config_preproc, config_train):
    raw = input_data.run(config_data)
    raw = feature_engineering.run(raw)
    sample_split = split.run(raw, config_data)
    attributes = preprocess.get_attributes(training_set=sample_split['training_set'],
                                           config_preproc=config_preproc)
    pipeline = preprocess.gen_full_pipeline(attributes, config_preproc)
    prepared_data = preprocess.run(training_set=sample_split['training_set'],
                                   labels=sample_split['training_labels'],
                                   full_pipeline=pipeline)
    return {'split': sample_split,
            'attributes': attributes,
            'prepared_data': prepared_data,
            'pipeline': pipeline,
            'model': train.run(prepared_data,
                               labels=sample_split['training_labels'],
                               training_algo=config_train['algorithm'])}


if __name__ == "__main__":
    trained = run(config_data=config.CONFIG_DATA,
                  config_preproc=config.CONFIG_PREPROC,
                  config_train=config.CONFIG_TRAIN)
    lib_io.persist_model(trained['model'], config.CONFIG_OUTPUT['path'])
    model_evaluation = evaluate.run(trained,
                                    config_evaluation=config.CONFIG_EVALUATION,
                                    config_preproc=config.CONFIG_PREPROC)
