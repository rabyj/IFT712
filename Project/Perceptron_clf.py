from classifier import classifier
from sklearn.linear_model import Perceptron # https://scikit-learn.org/stable/modules/linear_model.html#perceptron

class Perceptron_clf(classifier):
    """[summary]

    Args:
        classifier (Perceptron)
    """
    def __init__(self, X_train, X_test, t_train, t_test):
        """[summary]

        Args:
            X_train (np.array)
            X_test (np.array)
            t_train (np.array)
            t_test (np.array)
        """

        super(Perceptron_clf, self).__init__(X_train, X_test, t_train, t_test)
        self.model = "Perceptron"
        self.classifier = Perceptron(penalty="l2",max_iter=1000)
        self.parameters = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                           'eta0' : [1e-5, 2e-4, 1e-4, 2e-3, 1e-3]}
