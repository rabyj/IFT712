from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

class Classifier:
    """Super class to use different sklearn classifiers.

    - Optimizes hyper-parameters and display results
    - Computes accuracy and f1-score on given datasets
    - Make predictions on new data.

    Args :
        X_train (np.array) : training features
        t_train (np.array) : training labels

    Attributes :
        X_train (np.array)
        t_train (np.array)
        grid_clf (sklearn GridSearchCV object) : Fitted estimator and grid-search results
        classifier (sklearn classifier) : Used classifier
        hyperparams (dict) : Hyperparameters ranges for CV/grid search (can be single values)
        model_name (str) : the name of the classifier
    """
    def __init__(self, X_train, t_train):
        """Initialize all attributes."""
        self.X_train = X_train
        self.t_train = t_train

        self.grid_clf = None
        self.classifier = None
        self.hyperparams = {}
        self.model_name = None


    def optimize_hyperparameters(self, n_fold=8, metric="accuracy"):
        """Find the best parameters for the classifier through grid-search and
        StratifiedKFold cross-validation.

        Computes accuracy and f1-score on validation sets, optimizes on given metric.
        Only "f1_macro" and "accuracy" metrics are currently supported.

        Retrains the classifier on the whole training dataset afterwards.

        Uses all available processors.

        Args:
            n_fold (int) : Number of folds for StratifiedKFold.
            metric (string) : Metric to optimize on.
        """
        scores=["accuracy", "f1_macro"]
        grid = GridSearchCV(
            self.classifier, self.hyperparams, scoring=scores, n_jobs=-1, verbose=0, cv=n_fold, refit=metric
            )
        grid.fit(self.X_train, self.t_train)
        self.grid_clf = grid


    def new_train(self, X, t):
        """Train the best found estimator on a new X and t."""
        self.grid_clf.best_estimator_.fit(X, t)


    def get_accuracy(self, X, t):
        """Return the accuracy score on data X and labels t from optimized estimator.

        Args:
            X (np.array)
            t (np.array)

        Returns:
            accuracy (float)
        """
        return accuracy_score(t, self.grid_clf.best_estimator_.predict(X))


    def get_f1_score(self, X, t):
        """Return the macro f1-score on data X and labels t from optimized estimator.

        Args:
            X (np.array)
            t (np.array)

        Returns:
            f1_score (float)
        """
        return f1_score(t, self.grid_clf.best_estimator_.predict(X), average="macro")


    def predict(self, data):
        """Classify new data points.

        Use preprocessor to transform back into string labels if needed.

        Returns:
            array of integer labels
        """
        return self.grid_clf.best_estimator_.predict(data)


    def get_general_validation_results(self):
        """Return accuracy and f1-score (with std) on validation sets.

        Returns:
            valid_acc (float)
            valid_acc_std (float)
            valid_f1 (float)
            valid_f1_std (float)
        """
        results = self.grid_clf.cv_results_
        i = self.grid_clf.best_index_

        valid_acc = results["mean_test_accuracy"][i]
        valid_acc_std = results["std_test_accuracy"][i]

        valid_f1 = results["mean_test_f1_macro"][i]
        valid_f1_std = results["std_test_f1_macro"][i]

        return valid_acc, valid_acc_std, valid_f1, valid_f1_std


    def display_general_validation_results(self):
        """Display optimised results with 95% confidence interval (2sigma)"""

        acc, acc_std, f1, f1_std = self.get_general_validation_results()

        print("-------------------------------------------------------")
        print("Validation results")
        print("The model : {}".format(self.model_name))
        print("The best parameters : {}".format(self.grid_clf.best_params_))
        print("Mean accuracy and macro f1-score with 2 sigma interval on validation sets")
        print("Accuracy: {:0.3f}+/-{:0.03f}".format(acc, acc_std*2))
        print("f1-score: {:0.3f}+/-{:0.03f}".format(f1, f1_std*2))
        print("-------------------------------------------------------\n")


    def _create_results_df(self):
        """Return pandas dataframe of cross-validation results with certain columns."""
        chosen_columns = (
            ["param_" + key for key in self.grid_clf.cv_results_["params"][0].keys()] +
            ["mean_test_accuracy", "std_test_accuracy", "mean_test_f1_macro", "std_test_f1_macro"]
        )
        df = pd.DataFrame.from_dict(self.grid_clf.cv_results_)
        return df[chosen_columns]


    def return_cv_results(self):
        """Return pandas dataframe of cross-validation results."""
        return self._create_results_df()


    def display_cv_results(self):
        """Display grid-search and cross-validation results."""
        print("-------------------------------------------------------")
        print("The model : {}".format(self.model_name))
        print("All grid-search validation results.")
        df = self._create_results_df()
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None,
            "display.width", None, "display.max_colwidth", -1
        ):
            print(df)
        print("-------------------------------------------------------\n")
