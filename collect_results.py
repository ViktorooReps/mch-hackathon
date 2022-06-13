import pickle

import numpy as np

if __name__ == '__main__':
    files = [
        'cached_0_300.pkl',
        'cached_300_600.pkl',
        'cached_600_900.pkl',
        'cached_900_1800.pkl',
        'cached_1800_2700.pkl',
        'cached_2700_3600.pkl',
        'cached_3600_4500.pkl',
        'cached_4500_5400.pkl',
        'cached_5400_6300.pkl',
        'cached_6300_7200.pkl'
    ]

    all_X = []
    all_y = []

    for file in files:
        with open(file, 'rb') as f:
            X, y = pickle.load(f)
            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(X.shape)
    print(y.shape)
