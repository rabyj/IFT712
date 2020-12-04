import numpy as np
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from classifier import Classifier

class RF_clf(Classifier):
    """[summary]

    Args:
        classifier (RandomForestClassifier)
    """

    def __init__(self, X_train, X_valid, t_train, t_valid):
        """[summary]

        Args:
            X_train (np.array)
            X_valid (np.array)
            t_train (np.array)
            t_valid (np.array)
        """

        super(RF_clf, self).__init__(X_train, X_valid, t_train, t_valid)
        self.model_name = "Random Forest Classifier"
        self.classifier = RandomForestClassifier()
        self.parameters = {"n_estimators": np.arange(85, 100),
                           "max_depth": np.linspace(40, 100, num = 10)}
