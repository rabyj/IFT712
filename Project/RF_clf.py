import numpy as np
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from classifier import Classifier

class RF_clf(Classifier):
    """Random Forest classifier

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super(RF_clf, self).__init__(X_train, t_train)
        self.model_name = "Random Forest"
        self.classifier = RandomForestClassifier()
        self.hyperparams = {"n_estimators": np.arange(85, 100),
                           "max_depth": np.linspace(40, 100, num = 10)}
