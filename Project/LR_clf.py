from sklearn.linear_model import LogisticRegression # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

from classifier import Classifier

class LR_clf(Classifier):
    """Logistic Regression classifier

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super(LR_clf, self).__init__(X_train, t_train)
        self.model_name = "Logistic Regression"
        self.classifier = LogisticRegression(solver="saga", multi_class="multinomial", max_iter=1000)
        self.set_hyperparams()


    def set_hyperparams(self, penalty="elasticnet", C=8.9, l1_ratio=1/3):
        """Set hyperparameters with single values. See sklearn doc for meaning.

        Default values are for scaled data with PCA components explaining 80% of variance.
        """
        self.hyperparams["penalty"] = [penalty]
        self.hyperparams["C"] = [C]
        self.hyperparams["l1_ratio"] = [l1_ratio]


    def set_hyperparams_range(self, penalty, C, l1_ratio):
        """Set hyperparameters with ranges. See sklearn doc for meaning.

        Training will fail if any parameter is not list-like.
        """
        self.hyperparams["penalty"] = penalty
        self.hyperparams["C"] = C
        self.hyperparams["l1_ratio"] = l1_ratio
