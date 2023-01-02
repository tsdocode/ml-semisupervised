import numpy as np
import mlflow
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset.ssl_dataset import BaseSSLDataset
from model.graph_hamonic import HarmonicFunctionSSL

from .ssl_experiment import Experiemnt


class HarmonicExperiment(Experiemnt):
    def __init__(self, datasets: List[BaseSSLDataset], method="GraphHarmonicFunction") -> None:
        super().__init__(method, datasets, models="")
        self.label_rates = [.01, .05, .1, .2, .5]


    def run(self):
        mlflow.sklearn.autolog()


        for dataset in self.datasets:
            n_classes = len(np.unique(dataset.y))
            with mlflow.start_run(run_name=f"{self.method}-{dataset.name}"):
                for rate in self.label_rates:
                    n_sample = int(len(dataset.X) * rate)
                    X_train, y_train, X_test, y_test = dataset.get_data(n_sample)

                    harmonic = HarmonicFunctionSSL(X_train, y_train)
                    pred = harmonic.inference(X_test, y_test)

                    mlflow.log_metric(f"Accuracy - {rate}", accuracy_score(pred, y_test))
                    mlflow.log_metric(f"Precision - {rate}", precision_score(pred, y_test, average='micro'))
                    mlflow.log_metric(f"Recall - {rate}", recall_score(pred, y_test, average='micro'))
                    mlflow.log_metric(f"F1 - {rate}", f1_score(pred, y_test, average='micro'))
    