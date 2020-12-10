import numpy as np
from sklearn.linear_model import LogisticRegression # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

from classifier import Classifier

class LR_clf(Classifier):
    """[summary]

        classifier (LogisticRegression)
    """

    def __init__(self, X_train, t_train):
        """[summary]

        Args:
            X_train (np.array)
            t_train (np.array)
        """

        super(LR_clf, self).__init__(X_train, t_train)
        self.model_name = "Logistic Regression"
        self.classifier = LogisticRegression(solver="saga", multi_class="multinomial", max_iter=1000)
        # self.hyperparams = {
        #     "penalty" : ["elasticnet"],
        #     "C" : np.linspace(0.1, 10, num=10),
        #     "l1_ratio" : np.linspace(0, 1, num=10)
        #     }
        self.hyperparams = {
            "penalty" : ["elasticnet"],
            "C" : [8.9],
            "l1_ratio" : [1/3]
            }
