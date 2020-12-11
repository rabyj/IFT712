from sklearn.svm import SVC # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

from classifier import Classifier

class SVM_clf(Classifier):
    """Support Vector Machine classifier.

    Extends parent class with hyperparameters setter on top.

    See parent class "Classifier" docstring.
    """
    def __init__(self, X_train, t_train):
        """Calls parent class init and
        sets model_name, classifier and hyperparams attributes.
        """
        super(SVM_clf, self).__init__(X_train, t_train)
        self.model_name = "Support Vector Machine (SVM)"
        self.classifier = SVC(max_iter=10000)
        self.set_hyperparams()


    def set_hyperparams(self, kernel="rbf", C=5, gamma=0.003, degree=None, coef0=None):
        """Set hyperparameters with single values. See sklearn doc for meaning.

        Be careful, degree and coef0 are not used by certain kernels, but could
        still be listed in the results if set.

        Default values are for scaled data with no PCA.
        """
        self.hyperparams["kernel"] = [kernel]
        self.hyperparams["C"] = [C]
        self.hyperparams["gamma"] = [gamma]
        if degree is not None:
            self.hyperparams["degree"] = [degree]
        if coef0 is not None:
            self.hyperparams["coef0"] = [coef0]


    def set_hyperparams_range(self, kernel, C, gamma, degree=None, coef0=None):
        """Set hyperparameters with ranges. See sklearn doc for meaning.

        Be careful, degree and coef0 are not used by certain kernels,
        and useless recalculations are possible if set.

        Training will fail if any parameter is not list-like.
        """
        self.hyperparams["kernel"] = kernel
        self.hyperparams["C"] = C
        self.hyperparams["gamma"] = gamma
        if degree is not None:
            self.hyperparams["degree"] = degree
        if coef0 is not None:
            self.hyperparams["coef0"] = coef0
