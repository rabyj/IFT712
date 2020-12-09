import warnings
#warnings.filterwarnings("ignore")
from tqdm import tqdm

from preprocessing import Preprocessor
from LR_clf import LR_clf
from Perceptron_clf import Perceptron_clf
from SVM_clf import SVM_clf
from MLP_clf import MLP_clf
from RF_clf import RF_clf
from NB_clf import NB_clf

import numpy as np

def main():
    """Run general parameter search and print results for each classifier."""

    preprocessor = Preprocessor()
    preprocessor.import_data("data/train.csv")

    # erase non-transformed data
    X_train, t_train = preprocessor.encode_labels(use_new_encoder=True)
    X_train = preprocessor.scale_data(X_train, use_new_scaler=True)
    X_train = preprocessor.apply_pca(X_train, use_new_pca=True)

    # classifiers = [LR_clf, Perceptron_clf, SVM_clf, MLP_clf, RF_clf, NB_clf]
    classifiers = [LR_clf]
    clfs = []

    for clf in classifiers:
        clfs.append(clf(X_train, t_train))


    for i in tqdm(range(len(classifiers))):

        clfs[i].optimize_hyperparameters()
        clfs[i].display_general_results()
        clfs[i].display_cv_results()

if __name__ == "__main__":
    main()
