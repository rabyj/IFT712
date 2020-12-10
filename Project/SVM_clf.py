import numpy as np
from sklearn.svm import SVC # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

from classifier import Classifier

class SVM_clf(Classifier):
    """Support Vector Machine classifier.

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super(SVM_clf, self).__init__(X_train, t_train)
        self.model_name = "Support Vector Machine (SVM)"
        self.classifier = SVC(max_iter=10000)
        self.hyperparams = {
            "kernel": ["poly", "rbf", "sigmoid"],
            "C": np.linspace(1, 5, num=5),
            "gamma": np.geomspace(1e-3, 0.01, num=5),
            "degree": np.arange(2, 5, 2)
            }
