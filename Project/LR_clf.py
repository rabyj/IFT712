from classifier import classifier
from sklearn.linear_model import LogisticRegression # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

class LR_clf(classifier):
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
        
        super(LR_clf, self).__init__(X_train, X_test, t_train, t_test)
        self.model = "LogisticRegression"
        self.classifier = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=10000)
        self.parameters = {'penalty' : ['l1','l2']}