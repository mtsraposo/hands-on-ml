from sklearn.datasets import fetch_openml
import numpy as np


def run():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target']
    y = y.astype(np.uint8)
    # MNIST is already split into a training set (first 60,000 images) and
    # a test set (last 10,000 images). In addition, the training set is already shuffled,
    # so no other transformations are necessary.
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return {'data': {'train': X_train,
                     'test': X_test},
            'target': {'train': y_train,
                       'test': y_test}}
