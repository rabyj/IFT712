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
        self.classifier = Perceptron(max_iter=1000)
        self.set_hyperparams()


    def set_hyperparams(self, penalty="l1", alpha=0.001, eta0=0.1):
        """Set hyperparameters with single values. See sklearn doc for meaning.

        Default values are for scaled data with no PCA.
        """
        self.hyperparams["penalty"] = [penalty]
        self.hyperparams["alpha"] = [alpha]
        self.hyperparams["eta0"] = [eta0]


    def set_hyperparams_range(self, penalty, alpha, eta0):
        """Set hyperparameters with ranges. See sklearn doc for meaning.

        Training will fail if any parameter is not list-like.
        """
        self.hyperparams["penalty"] = penalty
        self.hyperparams["alpha"] = alpha
        self.hyperparams["eta0"] = eta0
