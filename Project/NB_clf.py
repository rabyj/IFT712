import numpy as np
from sklearn.naive_bayes import GaussianNB # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

from classifier import Classifier

class NB_clf(Classifier):
    """[summary]

        classifier (LogisticRegression)
    """

    def __init__(self, X_train, t_train):
        """[summary]

        Args:
            X_train (np.array)
            t_train (np.array)
        """

        super(NB_clf, self).__init__(X_train, t_train)
        self.model_name = "Gaussian Naive Bayes"
        self.classifier = GaussianNB()
        self.hyperparams = {"var_smoothing": np.arange(1e-8, 1e-5, 1e-7)}
