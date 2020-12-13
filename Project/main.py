import warnings
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from preprocessing import Preprocessor
from LR_clf import LR_clf
from Perceptron_clf import Perceptron_clf
from SVM_clf import SVM_clf
from MLP_clf import MLP_clf
from RF_clf import RF_clf
from NB_clf import NB_clf

def test_PCA(preprocessor, X_train_scaled, t_train):
    """Code to create some figure for PCA results."""
    step = 5
    n_components_range = range(15, 64*3, step)
    scores = []
    for n_components in n_components_range:

        X_train = preprocessor.apply_pca(X_train_scaled, use_new_pca=True, n_components=n_components, whiten=False)

        clf = Perceptron_clf(X_train, t_train)
        clf.optimize_hyperparameters()
        acc, acc_std, _, _ = clf.get_general_validation_results()
        scores.append((acc, acc_std))

    plt.figure()
    scores = np.array(scores)

    i = np.argmax(scores[:,0])
    best_acc, std = scores[i]
    print("See fig. Best result is {:.03f}+-{:.03f} at {} components".format(best_acc, std, i*step+n_components_range[0]))

    plt.errorbar(n_components_range, scores[:,0] , yerr=scores[:,1], marker='o')
    plt.xlabel("Number of PCA components")
    plt.ylabel(r"Accuracy$\pm1\sigma$")
    plt.title(r"Performance on validation sets for different PCA. Whiten=False.")
    plt.savefig("Perceptron_pca_no_whiten.png")


def make_new_predictions(path, preprocessor, clf):
    """Import future unlabeled data and make predictions on it"""
    new_data = preprocessor.import_unlabeled_data(path)
    new_data = preprocessor.scale_data(new_data)
    new_data = preprocessor.apply_pca(new_data)
    int_predictions = clf.predict(new_data)
    label_predictions = preprocessor.invert_encoding(int_predictions)
    return label_predictions


def main():
    """Run general parameter search and print results for each classifier."""

    preprocessor = Preprocessor()
    preprocessor.import_labeled_data("data/train.csv")

    X_total, t_total = preprocessor.encode_labels(use_new_encoder=True)
    X_train, X_test, t_train, t_test = preprocessor.train_test_split(X_total, t_total)

    # transform data and overwrite non-transformed data
    X_train_scaled = preprocessor.scale_data(X_train, use_new_scaler=True)
    X_test_scaled = preprocessor.scale_data(X_test, use_new_scaler=False)

    # apply PCA
    X_train = preprocessor.apply_pca(X_train_scaled, use_new_pca=True, n_components=27, whiten=False)
    X_test = preprocessor.apply_pca(X_test_scaled, use_new_pca=False)

    clf = NB_clf(X_train, t_train)
    clf.optimize_hyperparameters()
    clf.display_general_validation_results()
    clf.display_cv_results()
    print("Test accuracy : {:.03f}".format(clf.get_accuracy(X_test, t_test)))
    print("Test f1-score : {:.03f}".format(clf.get_f1_score(X_test, t_test)))

    label_predictions = make_new_predictions("data/test.csv", preprocessor, clf)


if __name__ == "__main__":
    main()
