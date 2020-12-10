from sklearn.linear_model import Perceptron # https://scikit-learn.org/stable/modules/linear_model.html#perceptron

from classifier import Classifier

class Perceptron_clf(Classifier):
    """Perceptron classifier

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super(Perceptron_clf, self).__init__(X_train, t_train)
        self.model_name = "Perceptron"
        self.classifier = Perceptron(penalty="l2",max_iter=1000)
        self.parameters_range = {"alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                           "eta0" : [1e-5, 2e-4, 1e-4, 2e-3, 1e-3]}
