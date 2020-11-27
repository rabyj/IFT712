from classifier import classifier
from sklearn.naive_bayes import GaussianNB # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html


class NB_clf(classifier):
    """[summary]

    Args:
        classifier ([type]): [description]
    """
    def __init(self, X_train, X_test, t_train, t_test):
        """[summary]

        Args:
            X_train ([type]): [description]
            X_test ([type]): [description]
            t_train ([type]): [description]
            t_test ([type]): [description]
        """
        super(NB_clf, self).__init__(X_train, X_test, t_train, t_test)
        self.model = "naive_bayes GaussianNB"
        self.classifier = GaussianNB()
        self.parameters = {'var_smoothing': [1e-11, 1e-10, 1e-09, 1e-08, 1e-07]}