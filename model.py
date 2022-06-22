"""Model libraries"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import mutual_info_score, confusion_matrix, recall_score, f1_score

class Model:
    def __init__(self,penalty, alpha, max_iter=50000):
        self.model = SGDClassifier(penalty=penalty,
                                   alpha=alpha,
                                   max_iter=max_iter)#LogisticRegression(max_iter=max_iter)


    def fit(self, X, Y):
        self.model.fit(X, Y)

    def partial_fit(self, X, Y):
        self.model.partial_fit(X,Y)

    def get_params(self):
        return {"coef": self.model.coef_,
                        "intercept": self.model.intercept_,
                        "classes": self.model.classes_}

    def set_params(self, model_params):
        """-----OLD----
        prev_model_params = self.load("model_params")
        classes = prev_model_params["classes"]

        ind = [np.where(model_params["classes"] == value)[0][0] for value in classes]
        model_params["coef"] = model_params["coef"][ind]
        model_params["intercept"] = model_params["intercept"][ind]
        model_params["classes"] = model_params["classes"][ind]
        ---------------"""
        self.model.coef_ = model_params["coef"]
        self.model.intercept_ = model_params["intercept"]
        #self.model.classes_ = model_params["classes"]

    def predict(self, X):
        return self.model.predict(X)
    """Measure performance on test set and return:
            -Mutual information score
            -Confusion Matrix"""
    def measure_performance(self, X, Y):
        Y_pred = self.model.predict(X)
        mi = mutual_info_score(Y, Y_pred)
        conf_matrix = confusion_matrix(Y, Y_pred)
        recall = recall_score(Y, Y_pred)
        f1 = f1_score(Y, Y_pred)

        return mi, conf_matrix, recall, f1

