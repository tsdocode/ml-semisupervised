import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split


class BaseSSLDataset:
    def __init__(self) -> None:
        self.X = []
        self.y = []

    def get_data(self, num_labels):
        assert num_labels < len(self.X)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=True
        )


        num_class = len(np.unique(self.y))

        samples_per_class = int(num_labels / num_class)

        new_X, new_y = [], []

        idxes = []

        for c in range(num_class):
            idx = np.where(y_train == c)[0]
            idx = np.random.choice(idx, min(samples_per_class, len(idx)), False)

            idxes.extend(idx)

        new_X = deepcopy(X_train)
        new_y = deepcopy(y_train)

        for i in range(len(new_y)):
            if i not in idxes:
                new_y[i] = -1

        return new_X, new_y, X_test, y_test
