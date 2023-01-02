from .ssl_experiment import Experiemnt
from copy import deepcopy
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class MultiviewTrainingExperiment(Experiemnt):
    def __init__(self, datasets, estimators, method="Co-Multiview-Training") -> None:
        super().__init__(method, datasets, estimators)
        self.estimators = estimators
        

    
    def multiview_predict_prob(self, Xs, models):
        X1 = Xs[0]
        X2 = Xs[1]

        y1_probs =  models[0].predict_proba(X1)
        y2_probs =  models[1].predict_proba(X2)
        
        return (y1_probs + y2_probs) * .5

    def multiview_predict(self, Xs, models):
        X1 = Xs[0]
        X2 = Xs[1]


        # predict each view independently
        y1 = models[0].predict(X1)
        y2 = models[1].predict(X2)

        # initialize
        y_pred = np.zeros(X1.shape[0],)

        # predict samples based on trained classifiers
        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
            # if classifiers agree, use their prediction
            if y1_i == y2_i:
                y_pred[i] = y1_i
            # if classifiers don't agree, take the more confident
            else:
                y1_probs =  models[0].predict_proba([X1[i]])[0]
                y2_probs =  models[1].predict_proba([X2[i]])[0]
                sum_y_probs = [prob1 + prob2 for (prob1, prob2) in
                            zip(y1_probs, y2_probs)]
                max_sum_prob = max(sum_y_probs)
                y_pred[i] = models[0].classes_[sum_y_probs.index(max_sum_prob)]

        return y_pred

    
    def multiview_training(self, views, labels, models, k_best, Xs_test, y_test, label_rate):
        labels = deepcopy(labels)
        has_label = labels != -1 

        print(has_label.shape)

        # print(has_label)
        best_models = ""
        best_score = 0
        iter_count = 0
        score = 0

        while not np.all(has_label):
            iter_count += 1


            for i, model in enumerate(models):
                model.fit(views[i][has_label], labels[has_label])



            unlabed_views = [view[~has_label] for view in views]

    #             # Predict unlabeled of other views
            prob = self.multiview_predict_prob(
                unlabed_views, models
            )


            selected_prob = np.max(prob, axis =1)
            pred = model.classes_[np.argmax(prob, axis = 1)]

            n_to_select = min(k_best, selected_prob.shape[0])

            if n_to_select == selected_prob.shape[0]:
                selected = np.ones_like(selected_prob, dtype=bool)
            else:
                selected = np.argpartition(-selected_prob, n_to_select)[:n_to_select]

            selected_full = np.nonzero(~has_label)[0][selected]


            labels[selected_full] = pred[selected]
            has_label[selected_full] = True


            test_pred = self.multiview_predict(Xs_test, models)

            if best_models == "":
                best_models = models
            else:
                score = accuracy_score(test_pred, y_test)
                mlflow.log_metric(f"Accuracy - {label_rate}", score)
                if  score > best_score:
                    best_score = score
                    mlflow.log_metric(f"Best Accuracy - {label_rate}", best_score)
                    mlflow.log_metric(f"Precision - {label_rate}", precision_score(test_pred, y_test, average='micro'))
                    mlflow.log_metric(f"Recall - {label_rate}", recall_score(test_pred, y_test, average='micro'))
                    mlflow.log_metric(f"F1 - {label_rate}", f1_score(test_pred, y_test, average='micro'))
                    
                    best_models = deepcopy(models)
      


        return best_models    


    def run(self):
        model_names = [type(model).__name__ for model in self.estimators]
        for i, dataset in enumerate(self.ssl_datasets):
            run_name = f"{self.method}-{self.dataset_names[i]}-{'-'.join(model_names)}"
            with mlflow.start_run(run_name=run_name):
                for rate in self.label_rates:
                    print(f"Start experiment with label rate {rate}")
                    model = deepcopy(model)
                    dataset_view_1 = dataset[rate][0]

                    dataset_view_2 = np.array([self.datasets[i].make_noise(item) + item for item in dataset[rate][0]])
                    X_test = dataset[rate][2]
                    X_test_noise = [self.datasets[i].make_noise(item) + item for item in X_test]
                    views = [dataset_view_1, dataset_view_2]

                    result = self.multiview_training(
                        views, dataset[rate][1], self.estimators, 15, [X_test, X_test_noise], dataset[rate][3], rate
                    )
