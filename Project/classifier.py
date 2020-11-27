# processing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

class classifier:
    """[summary]
    
    """
    def __init__(self, X_train, X_test, t_train, t_test):
        """[summary]

        Args:
            X_train ([type]): [description]
            X_test ([type]): [description]
            t_train ([type]): [description]
            t_test ([type]): [description]
        """
        self.X_train = X_train
        self.X_test = X_test
        self.t_train = t_train
        self.t_test = t_test
        
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.classifier = None
        self.parameters = {'var_smoothing': [1e-11, 1e-10, 1e-09, 1e-08, 1e-07]}
        self.model = "classifier"
        
    def getHyperParameters(self):
        """[summary]
        """
        
        grid = GridSearchCV(self.classifier, self.parameters, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(self.X_train, self.t_train)

        self.best_estimator_ = grid.best_estimator_
        self.best_score_ = grid.best_score_
        self.best_params_ = grid.best_params_
    
    def trainDataset(self):
        """[summary]
        """
        self.best_estimator_.fit(self.X_train, self.t_train)
    
    def getAccuracyScore(self, x, t):
        """[summary]

        Args:
            x ([type]): [description]
            t ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        return accuracy_score(t, self.best_estimator_.predict(x))
    
    def getF1Score(self, x, t):
        """[summary]

        Args:
            x ([type]): [description]
            t ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        return f1_score(t, self.best_estimator_.predict(x), average="weighted")

    def displayResults(self):
        """[summary]
        """
        print("-------------------------------------------------------\n")
        print("The model : "+ self.model)
        print("The best parameters : {}".format(self.best_params_))
        print("Training accuracy: {}".format(self.getAccuracyScore(self.X_train, self.t_train)))
        print("Test accuracy: {}".format(self.getAccuracyScore(self.X_test, self.t_test)))
        print("Training f1-score: {}".format(self.getF1Score(self.X_train, self.t_train)))
        print("Test f1-score: {}".format(self.getF1Score(self.X_test, self.t_test)))
        print("-------------------------------------------------------\n")
        