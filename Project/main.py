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

if __name__ == "__main__":

    preprocessor = Preprocessor()
    X_train, t_train = preprocessor.import_encode()

    # classifiers = [LR_clf, Perceptron_clf, SVM_clf, MLP_clf, RF_clf, NB_clf]
    classifiers = [LR_clf]
    clfs = []

    for clf in classifiers:
        clfs.append(clf(X_train, t_train))


    for i in tqdm(range(len(classifiers))):

        clfs[i].optimize_hyperparameters()
        clfs[i].display_general_results()
        clfs[i].display_cv_results()
