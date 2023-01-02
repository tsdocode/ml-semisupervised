import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



from .ssl_experiment import Experiemnt
from sklearn.mixture import GaussianMixture as GMM

from dataset.ssl_dataset import BaseSSLDataset
from typing import List

class GMMExperiment(Experiemnt):
    def __init__(self, datasets: List[BaseSSLDataset], method="GMM") -> None:
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

                    means_init = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in range(n_classes)])

                    gmm = GMM(n_components=n_classes,
                    covariance_type="full", init_params='k-means++', max_iter=2000, means_init= means_init)

                    

                    # mlflow.log_param("gmm mean", gmm.means_)

                    gmm.fit(X_train, y_train)
                    pred = gmm.predict(X_test)

                    mlflow.log_metric(f"Accuracy - {rate}", accuracy_score(pred, y_test))
                    mlflow.log_metric(f"Precision - {rate}", precision_score(pred, y_test, average='micro'))
                    mlflow.log_metric(f"Recall - {rate}", recall_score(pred, y_test, average='micro'))
                    mlflow.log_metric(f"F1 - {rate}", f1_score(pred, y_test, average='micro'))




    