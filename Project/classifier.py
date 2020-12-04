"""Parent class for classifiers"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

class Classifier:
    """[summary]

       super class : - contains the variables that are used by the classifiers
                     - calculate the best hyper-parameters for the classifiers
                     - training the datasets
                     - calculate the accuracy of the classifier
                     - calculate the f1-score
                     - display the scores of the model
    """
    def __init__(self, X_train, t_train):
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
        self.t_train = t_train

        self.grid_clf = None
        self.classifier = None
        self.parameters_range = None
        self.model_name = None

    def optimize_hyperparameters(self, n_fold=5, metric="accuracy"):
        """Find the best parameters for the classifier through grid-search and StratifiedKFold cross-validation.

        Retrain the classifier on the whole training dataset afterwards.

        Args:
            n_fold (int) : Number of folds for StratifiedKFold.
            metric (string) : Scoring metric for the grid search.
        """
        grid = GridSearchCV(self.classifier, self.parameters_range, scoring=metric, n_jobs=-1, verbose=1, cv=n_fold, refit=True)
        grid.fit(self.X_train, self.t_train)
        self.grid_clf = grid

    def new_train(self, X, t):
        """Train the best found estimator on a new X and t.
        """
        self.grid_clf.best_estimator_.fit(X, t)

    def get_accuracy(self, X, t):
        """Get the best found estimator accuracy on data X and labels t.

        Args:
            X (np.array)
            t (np.array)

        Returns:
            accuracy [float]
        """
        return accuracy_score(t, self.grid_clf.best_estimator_.predict(X))

    def get_f1_score(self, X, t):
        """Get the best found estimator f1 score on data X and labels t.

        Args:
            X (np.array)
            t (np.array)

        Returns:
            f1_score [float]
        """
        return f1_score(t, self.grid_clf.best_estimator_.predict(X), average="weighted")

    def display_general_results(self):
        """Display optimised results."""
        print("-------------------------------------------------------")
        print("The model : {}".format(self.model_name))
        print("The best parameters : {}".format(self.grid_clf.best_params_))
        print("Global training accuracy: {}".format(self.get_accuracy(self.X_train, self.t_train)))
        print("Global training f1-score: {}".format(self.get_f1_score(self.X_train, self.t_train)))
        print("Accuracy score on validation sets : {:0.3f}+/-{:0.03f}".format(
            self.grid_clf.cv_results_['mean_test_score'][self.grid_clf.best_index_],
            self.grid_clf.cv_results_['std_test_score'][self.grid_clf.best_index_]*2
        ))
        print("-------------------------------------------------------\n")

    def display_cv_results(self):
        """Display grid-search and cross-validation results."""
        print("-------------------------------------------------------")
        print("The model : {}".format(self.model_name))
        print("All grid-search validation results")
        means = self.grid_clf.cv_results_['mean_test_score']
        stds = self.grid_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, self.grid_clf.cv_results_['params']):
            print("{:0.3f} (+/-{:0.03f}) for {}".format(mean, std*2, params))
        print("-------------------------------------------------------\n")
