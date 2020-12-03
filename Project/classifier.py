# processing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

class classifier:
    """[summary]

       super class : - contains the variables that are used by the classifiers
                     - calculate the best hyper-parameters for the classifiers
                     - training the datasets
                     - calculate the accuracy of the classifier
                     - calculate the f1-score 
                     - display the scores of the model
    """
    def __init__(self, X_train, X_valid, t_train, t_valid):
        """[summary]

        Args:
            X_train (np.array)
            X_valid (np.array)
            t_train (np.array)
            t_valid (np.array)
            best_estimator_ (classifier) : the best estimator with parameters chosen by GridSearch 
            best_score_ (float) : the best score 
            classifier (classifier) : the working classifier
            parameters (dict)
            model (str) : the name of the classifier
            
        """
        self.X_train = X_train
        self.X_valid = X_valid
        self.t_train = t_train
        self.t_valid = t_valid
        
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.classifier = None
        self.parameters = None
        self.model = "classifier"
        
    def getHyperParameters(self):
        """[summary]
        
        find the best parameters for the classifier
        """
        
        grid = GridSearchCV(self.classifier, self.parameters, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(self.X_train, self.t_train)

        self.best_estimator_ = grid.best_estimator_
        self.best_score_ = grid.best_score_
        self.best_params_ = grid.best_params_
    
    def trainDataset(self):
        """[summary]
        
        train the data
        """
        self.best_estimator_.fit(self.X_train, self.t_train)
    
    def getAccuracyScore(self, x, t):
        """[summary]

        Args:
            x (np.array)
            t (np.array)

        Returns:
            accuracy [float]
        """
        
        return accuracy_score(t, self.best_estimator_.predict(x))
    
    def getF1Score(self, x, t):
        """[summary]

        Args:
            x (np.array)
            t (np.array)

        Returns:
            f1_score [float]
        """
        
        return f1_score(t, self.best_estimator_.predict(x), average="weighted")

    def displayResults(self):
        """[summary]
        
        display the information
        """
        print("-------------------------------------------------------\n")
        print("The model : "+ self.model)
        print("The best parameters : {}".format(self.best_params_))
        print("Training accuracy: {}".format(self.getAccuracyScore(self.X_train, self.t_train)))
        print("Validation accuracy: {}".format(self.getAccuracyScore(self.X_valid, self.t_valid)))
        print("Training f1-score: {}".format(self.getF1Score(self.X_train, self.t_train)))
        print("Validation f1-score: {}".format(self.getF1Score(self.X_valid, self.t_valid)))
        print("-------------------------------------------------------\n")
        