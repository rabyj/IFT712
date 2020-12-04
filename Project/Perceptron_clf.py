from sklearn.linear_model import Perceptron # https://scikit-learn.org/stable/modules/linear_model.html#perceptron

from classifier import Classifier

class Perceptron_clf(Classifier):
    """[summary]

    Args:
        classifier (Perceptron)
    """
    def __init__(self, X_train, X_valid, t_train, t_valid):
        """[summary]

        Args:
            X_train (np.array)
            X_valid (np.array)
            t_train (np.array)
            t_valid (np.array)
        """

        super(Perceptron_clf, self).__init__(X_train, X_valid, t_train, t_valid)
        self.model_name = "Perceptron"
        self.classifier = Perceptron(penalty="l2",max_iter=1000)
        self.parameters = {"alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                           "eta0" : [1e-5, 2e-4, 1e-4, 2e-3, 1e-3]}
