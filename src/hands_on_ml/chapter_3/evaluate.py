from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def cross_validation(data, model):
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
