from classifier import classifier
from sklearn.neural_network import MLPClassifier # https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification

class MLP_clf(classifier):
    """[summary]

    Args:
        classifier (MLPClassifier)
    """
    def __init__(self, X_train, X_test, t_train, t_test):
        """[summary]

        Args:
            X_train (np.array)
            X_test (np.array)
            t_train (np.array)
            t_test (np.array)
        """
        
        super(MLP_clf, self).__init__(X_train, X_test, t_train, t_test)
        self.model = "MLPClassifier"
        self.classifier = MLPClassifier(max_iter=10000)
        self.parameters = {'hidden_layer_sizes': [(30,), (40,), (50,)],
                            'learning_rate_init': [1e-2, 5e-2, 5e-1],
                            'activation': ['relu', 'logistic']}