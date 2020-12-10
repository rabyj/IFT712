import numpy as np
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from classifier import Classifier

class RF_clf(Classifier):
    """[summary]

    Args:
        classifier (RandomForestClassifier)
    """

    def __init__(self, X_train, t_train):
        """[summary]

        Args:
            X_train (np.array)
            t_train (np.array)
        """

        super(RF_clf, self).__init__(X_train, t_train)
        self.model_name = "Random Forest Classifier"
        self.classifier = RandomForestClassifier()
        self.hyperparams = {"n_estimators": np.arange(85, 100),
                           "max_depth": np.linspace(40, 100, num = 10)}
