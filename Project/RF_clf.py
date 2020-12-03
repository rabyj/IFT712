import numpy as np
from classifier import classifier
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

class RF_clf(classifier):
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
        self.model = "RandomForestClassifier"
        self.classifier = RandomForestClassifier()
        self.parameters = {"n_estimators": np.arange(85, 100),
                           "max_depth": np.linspace(40, 100, num = 10)}
