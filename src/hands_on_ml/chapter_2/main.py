import logging

from src.hands_on_ml.chapter_2 import input_data, split, feature_engineering, \
    config, preprocess, train, evaluate, \
    lib_io

logging.basicConfig(level=logging.INFO, format='%(message)s')


def run(config_data, config_preproc, config_train):
    raw = input_data.run(config_data)
    with_features = feature_engineering.run(raw)
    sample_split = split.run(with_features, config_data)
    attributes = preprocess.get_attributes(training_set=sample_split['training_set'],
                                           config_preproc=config_preproc)
    preproc_pipeline = preprocess.gen_pipeline(attributes, config_preproc)
    return {'split': sample_split,
            'attributes': attributes,
            'model': train.run(sample_split, config_train, preproc_pipeline)}


if __name__ == "__main__":
    trained = run(config_data=config.CONFIG_DATA,
                  config_preproc=config.CONFIG_PREPROC,
                  config_train=config.CONFIG_TRAIN)
    lib_io.persist_model(trained['model'], config.CONFIG_OUTPUT['path'])
    model_evaluation = evaluate.run(trained,
                                    config_evaluation=config.CONFIG_EVALUATION,
                                    config_preproc=config.CONFIG_PREPROC)
