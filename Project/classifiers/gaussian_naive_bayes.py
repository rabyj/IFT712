from sklearn.naive_bayes import GaussianNB # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

from classifiers.classifier import Classifier

class NaiveBayes(Classifier):
    """Gaussian Naive Bayes classifier

    Works better with uncorrelated features, apply PCA on data beforehand.

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super().__init__(X_train, t_train)
        self.model_name = "Gaussian Naive Bayes"
        self.classifier = GaussianNB()
        self.set_hyperparams()


    def set_hyperparams(self, var_smoothing=1e-9):
        """Set hyperparameters with single values. See sklearn doc for meaning."""
        self.hyperparams["var_smoothing"] = [var_smoothing]


    def set_hyperparams_range(self, var_smoothing):
        """Set hyperparameters with ranges. See sklearn doc for meaning.

        Training will fail if any parameter is not list-like.
        """
        self.hyperparams["var_smoothing"] = var_smoothing
