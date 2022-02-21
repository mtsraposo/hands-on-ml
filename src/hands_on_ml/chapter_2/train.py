import sklearn
from sklearn.pipeline import Pipeline


def gen_full_pipeline(preproc_pipeline, config_train):
    model_type = getattr(sklearn, config_train['algorithm']['type'])
    regression_class = getattr(model_type, config_train['algorithm']['name'])

    full_pipeline = Pipeline([
        ('preprocessor', preproc_pipeline),
        ('model', regression_class(**config_train['algorithm']['params']))
    ])
    return full_pipeline


def run(sample_split, config_train, preproc_pipeline):
    full_pipeline = gen_full_pipeline(preproc_pipeline, config_train)
    return {'pipeline': full_pipeline,
            'model': full_pipeline.fit(sample_split['training_set'],
                                       sample_split['training_labels'])}
