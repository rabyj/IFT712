from classifier import classifier
from sklearn.linear_model import LogisticRegression # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

class LR_clf(classifier):
    """[summary]

        classifier (LogisticRegression)
    """

    def __init__(self, X_train, X_test, t_train, t_test):
        """[summary]

        Args:
            X_train (np.array)
            X_test (np.array)
            t_train (np.array)
            t_test (np.array)
        """

        super(LR_clf, self).__init__(X_train, X_test, t_train, t_test)
        self.model = "LogisticRegression"
        self.classifier = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=10000)
        self.parameters = {'penalty' : ['l2']}
