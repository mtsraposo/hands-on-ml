import sklearn
from sklearn.pipeline import Pipeline


def gen_pipelines(config_train):
    model_type = getattr(sklearn, config_train['algorithm']['type'])
    regression_class = getattr(model_type, config_train['algorithm']['name'])

    regression_pipeline = Pipeline([
        ('model', regression_class(**config_train['algorithm']['params']))
    ])

    search_class = getattr(sklearn.model_selection, config_train['grid_search']['name'])
    search_pipeline = search_class(regression_pipeline, **config_train['grid_search']['params'])

    return {'regression': regression_pipeline,
            'search': search_pipeline}


def run(sample_split, config_train, preproc_pipeline):
    pipelines = gen_pipelines(config_train)
    preproc_pipeline.fit(sample_split['training_set'], sample_split['training_labels'])
    training_set_prepared = preproc_pipeline.transform(sample_split['training_set'])
    return {'preproc': preproc_pipeline,
            'training_set_prepared': training_set_prepared,
            'regression': pipelines['regression'].fit(training_set_prepared,
                                                      sample_split['training_labels']),
            'search': pipelines['search'].fit(training_set_prepared,
                                              sample_split['training_labels'])
            }
