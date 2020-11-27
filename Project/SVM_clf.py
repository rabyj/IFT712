import numpy as np
from classifier import classifier
from sklearn.svm import SVC # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

class SVM_clf(classifier):
    """[summary]

    Args:
        classifier ([type]): [description]
    """
    
    def __init__(self, X_train, X_test, t_train, t_test):
        """[summary]

        Args:
            X_train ([type]): [description]
            X_test ([type]): [description]
            t_train ([type]): [description]
            t_test ([type]): [description]
        """
        
        super(SVM_clf, self).__init__(X_train, X_test, t_train, t_test)
        self.model = "SVM"
        self.classifier = SVC(max_iter=10000)
        self.parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                           'gamma': [1e-2, 1e-1, 5e-2, 5e-1],
                           'C': np.linspace(1, 5, num=5),
                           "gamma": np.geomspace(1e-3, 0.01, num=5),
                           "degree": np.arange(2, 5, 2)}