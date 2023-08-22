from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve


def cross_validation(data, model):
    """
    Calculates the cross validation accuracy of the classifier model on the data.
    Serves as a counter example for performance measurement of classifiers.
    """
    accuracy = []
    skfolds = StratifiedKFold(n_splits=3)
    for train_index, test_index in skfolds.split(data['X']['train'], data['y']['train']):
        clone_clf = clone(model)
        X_train_folds = data['X']['train'].iloc[train_index]
        y_train_folds = data['y']['train'].iloc[train_index]
        X_test_fold = data['X']['train'].iloc[test_index]
        y_test_fold = data['y']['train'].iloc[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        accuracy += [n_correct / len(y_pred)]
    return accuracy


def gen_confusion_matrix(data, model):
    y_train_pred = cross_val_predict(model,
                                     data['X']['train'],
                                     data['y']['train'],
                                     cv=3)
    return {'confusion_matrix': confusion_matrix(data['y']['train'], y_train_pred),
            'precision': precision_score(data['y']['train'], y_train_pred),
            'recall': recall_score(data['y']['train'], y_train_pred),
            'f1': f1_score(data['y']['train'], y_train_pred)}


def gen_recall_curve(data, model):
    y_scores = cross_val_predict(model, data['X']['train'], data['y']['train'],
                                 cv=3, method='decision_function')
    return precision_recall_curve(data['y']['train'], y_scores)
