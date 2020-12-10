from sklearn.neural_network import MLPClassifier # https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification

from classifier import Classifier

class MLP_clf(Classifier):
    """Multilayer perceptron classifier

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super(MLP_clf, self).__init__(X_train, t_train)
        self.model_name = "Multilayer Perceptron"
        self.classifier = MLPClassifier(max_iter=10000)
        self.hyperparams = {"hidden_layer_sizes": [(70,10), (100,20), (50,30)],
                            "learning_rate_init": [1e-2, 5e-2, 5e-1],
                            "activation": ["relu", "logistic"]}
