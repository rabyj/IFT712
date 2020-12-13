from sklearn.neural_network import MLPClassifier # https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification

from classifiers.classifier import Classifier

class MLP(Classifier):
    """Multilayer perceptron classifier

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super().__init__(X_train, t_train)
        self.model_name = "Multilayer Perceptron"
        self.classifier = MLPClassifier(max_iter=5000)
        self.set_hyperparams()


    def set_hyperparams(self, hidden_layer_sizes=100, activation="relu", solver="lbfgs", alpha=10):
        """Set hyperparameters with single values. See sklearn doc for meaning.

        Default values are for scaled data with PCA components explaining 90% of variance.
        """
        self.hyperparams["hidden_layer_sizes"] = [hidden_layer_sizes]
        self.hyperparams["activation"] = [activation]
        self.hyperparams["solver"] = [solver]
        self.hyperparams["alpha"] = [alpha]


    def set_hyperparams_range(self, hidden_layer_sizes, activation, solver, alpha):
        """Set hyperparameters with ranges. See sklearn doc for meaning.

        Training will fail if any parameter is not list-like.
        """
        self.hyperparams["hidden_layer_sizes"] = hidden_layer_sizes
        self.hyperparams["activation"] = activation
        self.hyperparams["solver"] = solver
        self.hyperparams["alpha"] = alpha
