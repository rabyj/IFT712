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



if __name__ == "__main__":


    preprocessor = Preprocessor()
    train, test, t = preprocessor.import_encode()
    X_train, X_valid, t_train, t_valid = preprocessor.train_valid_split(train, t)

    classifiers = [LR_clf, Perceptron_clf, SVM_clf, MLP_clf, RF_clf, NB_clf]
    clfs = []

    for clf in classifiers:
        clfs.append(clf(X_train, X_valid, t_train, t_valid))


    for i in tqdm(range(len(classifiers))):

        clfs[i].get_hyperparameters()
        clfs[i].train_dataset()
        clfs[i].display_results()
