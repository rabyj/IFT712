import numpy as np
from classifier import classifier
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

class RF_clf(classifier):
    """[summary]

    Args:
        classifier (RandomForestClassifier)
    """
    
    def __init__(self, X_train, X_test, t_train, t_test):
        """[summary]

        Args:
            X_train (np.array)
            X_test (np.array)
            t_train (np.array)
            t_test (np.array)
        """
        
        super(RF_clf, self).__init__(X_train, X_test, t_train, t_test)
        self.model = "RandomForestClassifier"
        self.classifier = RandomForestClassifier()
        self.parameters = {"n_estimators": np.arange(85, 90),
                           "max_depth": np.linspace(10, 70, num = 10)}