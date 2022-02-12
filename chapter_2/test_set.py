import numpy as np
from zlib import crc32


def split_train_test(data, test_ratio):
    """
    Unless the same random seed is used, every call will yield different
    training sets, which is not desirable.
    The split_train_test_by_id solves this issue, by checking the data
    against a hashed id column.
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return {'train': data.iloc[train_indices],
            'test': data.iloc[test_indices]}


def add_unique_id(data):
    data['id'] = data['longitude'] * 1000 + data['latitude']
    return data


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    """Split according to a hashed id column, which should remain consistent
    over time."""
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return {'train': data.loc[~in_test_set],
            'test': data.loc[in_test_set]}
