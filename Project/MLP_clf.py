from classifier import classifier
from sklearn.neural_network import MLPClassifier # https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification

class MLP_clf(classifier):
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
        
        super(MLP_clf, self).__init__(X_train, X_test, t_train, t_test)
        self.model = "MLPClassifier"
        self.classifier = MLPClassifier(max_iter=10000)
        self.parameters = {'hidden_layer_sizes': [(50,), (70,)],
                            'learning_rate_init': [1e-2, 1e-1],
                            'activation': ['relu', 'logistic'],
                            'solver': ['adam']}