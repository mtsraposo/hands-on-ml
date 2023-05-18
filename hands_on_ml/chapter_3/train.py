from sklearn.linear_model import SGDClassifier


def five_detector(X_train, y_train_5):
    """
    Introductory binary classifier, with classes '5' and 'not-5'.
    """
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    return sgd_clf
