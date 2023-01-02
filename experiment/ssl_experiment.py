from datetime import datetime


class Experiemnt:
    def __init__(self, method, datasets, models) -> None:
        self.datasets = datasets
        self.method = method
        self.estimators = models
        self.label_rates = [0.05, 0.1, 0.2, 0.5]

        self.init_dataset()

    def init_dataset(self):
        ssl_datasets = []
        self.dataset_names = []

        for dataset in self.datasets:
            self.dataset_names.append(dataset.name)
            ssl_dataset = {}
            dataset_length = len(dataset.X)
            for rate in self.label_rates:
                number_of_labels = int(dataset_length * rate)
                ssl_dataset[rate] = dataset.get_data(number_of_labels)

            ssl_datasets.append(ssl_dataset)
        self.ssl_datasets = ssl_datasets

    def run(self):
        pass
