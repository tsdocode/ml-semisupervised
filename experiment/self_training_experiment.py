from .ssl_experiment import Experiemnt
from copy import deepcopy
import mlflow
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SelfTrainingExperiment(Experiemnt):
    def __init__(self, method, datasets, estimators) -> None:
        super().__init__(method, datasets, estimators)
        self.estimators = estimators

    def self_training(
        self, base_model, X_train, y_train, X_test, y_test, label_rate, k_best=15
    ):
        X_train = deepcopy(X_train)
        y_train = deepcopy(y_train)

        print(X_train.shape)

        logs = {}

        has_label = y_train != -1

        mlflow.log_param(
            f"Number  of label - {label_rate}", np.count_nonzero(has_label == True)
        )
        mlflow.log_param(f"K best - {label_rate}", k_best)

        label_iter = np.full_like(y_train, -1)
        label_iter[has_label] = 0

        base_model = base_model
        best_model = ""

        iter_count = 0

        pbar = tqdm(total=np.count_nonzero(has_label != True))

        while not np.all(has_label):

            base_model.fit(X_train[has_label], y_train[has_label])

            prob = base_model.predict_proba(X_train[~has_label])

            selected_prob = np.max(prob, axis=1)

            pred = base_model.classes_[np.argmax(prob, axis=1)]

            n_to_select = min(k_best, selected_prob.shape[0])

            # print("NO SELECT" , n_to_select)z

            if n_to_select == selected_prob.shape[0]:
                selected = np.ones_like(selected_prob, dtype=bool)
            else:
                selected = np.argpartition(-selected_prob, n_to_select)[:n_to_select]

            selected_full = np.nonzero(~has_label)[0][selected]

            y_train[selected_full] = pred[selected]

            has_label[selected_full] = True

            if selected.shape[0] == 0:
                break

            test_acc = base_model.score(X_test, y_test)

            if iter_count == 0:
                supervised_acc = test_acc

            if best_model == "":
                best_model = deepcopy(base_model)
            elif test_acc > best_model.score(X_test, y_test):
                best_model = deepcopy(base_model)


            test_pred = base_model.predict(X_test)
            mlflow.log_metric(f"Accuracy - {label_rate}", test_acc)
            mlflow.log_metric(f"Supervised Accuracy - {label_rate}", supervised_acc)
            mlflow.log_metric(f"Precision - {label_rate}", precision_score(test_pred, y_test, average='micro'))
            mlflow.log_metric(f"Recall - {label_rate}", recall_score(test_pred, y_test, average='micro'))
            mlflow.log_metric(f"F1 - {label_rate}", f1_score(test_pred, y_test, average='micro'))
            

            logs[np.count_nonzero(has_label == True)] = test_acc

            if best_model != "":
                mlflow.log_metric(
                    f"Best model Accuracy - {label_rate}",
                    best_model.score(X_test, y_test),
                )
                # print(f"Best model score {best_model.score(X_test, y_test)}")
            iter_count += 1
            pbar.update(k_best)

        return best_model, logs

    def run(self):
        for model in self.estimators:
            model_name = type(model).__name__
            for i, dataset in enumerate(self.ssl_datasets):
                run_name = f"{self.method}-{self.dataset_names[i]}-{model_name}"
                with mlflow.start_run(run_name=run_name):
                    for rate in self.label_rates:
                        print(f"Start experiment with label rate {rate}")
                        model = deepcopy(model)
                        best_model, logs = self.self_training(
                            model,
                            dataset[rate][0],
                            dataset[rate][1],
                            dataset[rate][2],
                            dataset[rate][3],
                            rate,
                        )
