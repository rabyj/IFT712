import numpy as np
from sklearn.svm import SVC # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

from classifier import Classifier

class SVM_clf(Classifier):
    """[summary]

    Args:
        classifier (SVC)
    """

    def __init__(self, X_train, X_valid, t_train, t_valid):
        """[summary]

        Args:
            X_train (np.array)
            X_valid (np.array)
            t_train (np.array)
            t_valid (np.array)
        """

        super(SVM_clf, self).__init__(X_train, X_valid, t_train, t_valid)
        self.model_name = "Support Vector Machine (SVM)"
        self.classifier = SVC(max_iter=10000)
        self.parameters = {"kernel": ["poly", "rbf", "sigmoid"],
                           "gamma": [1e-2, 1e-1, 5e-2, 5e-1],
                           "C": np.linspace(1, 5, num=5),
                           "gamma": np.geomspace(1e-3, 0.01, num=5),
                           "degree": np.arange(2, 5, 2)}
