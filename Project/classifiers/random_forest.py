from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from classifiers.classifier import Classifier

class RandomForest(Classifier):
    """Random Forest classifier

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super().__init__(X_train, t_train)
        self.model_name = "Random Forest"
        self.classifier = RandomForestClassifier()
        self.set_hyperparams()


    def set_hyperparams(self, criterion="gini", n_estimators=500, max_depth=50):
        """Set hyperparameters with single values. See sklearn doc for meaning.

        Default values are for scaled data with no PCA.
        """
        self.hyperparams["criterion"] = [criterion]
        self.hyperparams["n_estimators"] = [n_estimators]
        self.hyperparams["max_depth"] = [max_depth]


    def set_hyperparams_range(self, criterion, n_estimators, max_depth):
        """Set hyperparameters with ranges. See sklearn doc for meaning.

        Training will fail if any parameter is not list-like.
        """
        self.hyperparams["criterion"] = criterion
        self.hyperparams["n_estimators"] = n_estimators
        self.hyperparams["max_depth"] = max_depth
