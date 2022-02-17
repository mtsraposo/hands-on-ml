import logging

from src.hands_on_ml.chapter_2 import input_data, split, feature_engineering, config, preprocessing, train, evaluate, \
    lib_io, fine_tuning

logging.basicConfig(level=logging.INFO, format='%(message)s')


def run(config_data, config_preproc, config_train):
    housing_raw = input_data.run(config_data)
    housing_with_features = feature_engineering.run(housing_raw)
    housing_training = split.run(housing_raw, config_data)
    housing_attributes = preprocessing.get_attributes(housing=housing_training['set'],
                                                      config_preproc=config_preproc)
    housing_pipeline = preprocessing.gen_full_pipeline(housing_attributes, config_preproc)
    housing_prepared = preprocessing.run(housing=housing_training['set'],
                                         full_pipeline=housing_pipeline)
    return {'training': housing_training,
            'attributes': housing_attributes,
            'prepared_data': housing_prepared,
            'pipeline': housing_pipeline,
            'model': train.run(housing_prepared,
                               housing_labels=housing_training['labels'],
                               method=config_train['method'])}


def evaluate_and_tune(housing_model, config_evaluation, config_tuning, config_preproc):
    return {'evaluation': evaluate.run(housing_model, config_evaluation),
            'grid_search': fine_tuning.run(housing_model, config_tuning, config_preproc)}


if __name__ == "__main__":
    housing_model = run(config_data=config.CONFIG_DATA,
                        config_preproc=config.CONFIG_PREPROC,
                        config_train=config.CONFIG_TRAIN)
    lib_io.persist_model(housing_model['model'], config.CONFIG_OUTPUT['path'])
    model_evaluation = evaluate_and_tune(housing_model,
                                         config_evaluation=config.CONFIG_EVALUATION,
                                         config_tuning=config.CONFIG_TUNING,
                                         config_preproc=config.CONFIG_PREPROC)
