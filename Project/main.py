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

def main():
    """Run general parameter search and print results for each classifier."""

    preprocessor = Preprocessor()
    preprocessor.import_data("data/train.csv")

    X_total, t_total = preprocessor.encode_labels(use_new_encoder=True)
    X_train, X_test, t_train, t_test = preprocessor.train_test_split(X_total, t_total)

    # transform data and overwrite non-transformed data
    X_train_scaled = preprocessor.scale_data(X_train, use_new_scaler=True)
    X_test_scaled = preprocessor.scale_data(X_test, use_new_scaler=False)

    n_components_range = range(15, 50, 1)
    scores = []
    for n_components in n_components_range:

        X_train = preprocessor.apply_pca(X_train_scaled, use_new_pca=True, n_components=n_components, whiten=True)
        X_test = preprocessor.apply_pca(X_test_scaled, use_new_pca=False)

        # choose classifiers
        # classifiers = [LR_clf, Perceptron_clf, SVM_clf, MLP_clf, RF_clf, NB_clf]
        classifier_types = [NB_clf]
        clfs = []

        # init classifiers
        for clf in classifier_types:
            clfs.append(clf(X_train, t_train))

        # train
        for clf in clfs:

            clf.optimize_hyperparameters()
            acc, acc_std, _, _ = clf.get_general_validation_results()
            scores.append((acc, acc_std))
            # clf.display_cv_results()
            # print("Test accuracy : {:.03f}".format(clf.get_accuracy(X_test, t_test)))
            # print("Test f1-score : {:.03f}".format(clf.get_f1_score(X_test, t_test)))


    plt.figure()
    scores = np.array(scores)

    i = np.argmax(scores[:,0])
    best_acc, std = scores[i]
    print("See fig. Best result is {:.03f}+-{:.03f} at {} components".format(best_acc, std, i+15))

    plt.errorbar(n_components_range, scores[:,0] , yerr=scores[:,1], marker='o')
    plt.xlabel("Number of PCA components")
    plt.ylabel(r"Accuracy$\pm1\sigma$")
    plt.title(r"Performance on validation sets for different PCA. Whiten=True.")
    plt.savefig("NB_pca_w_withen_zoom.png")

if __name__ == "__main__":
    main()
