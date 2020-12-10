import warnings
from tqdm import tqdm

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
    X_train = preprocessor.scale_data(X_train, use_new_scaler=True)
    X_train = preprocessor.apply_pca(X_train, use_new_pca=True)

    X_test = preprocessor.scale_data(X_test, use_new_scaler=False)
    X_test = preprocessor.apply_pca(X_test, use_new_pca=False)

    # classifiers = [LR_clf, Perceptron_clf, SVM_clf, MLP_clf, RF_clf, NB_clf]
    classifiers = [LR_clf]
    clfs = []

    for clf in classifiers:
        clfs.append(clf(X_train, t_train))


    for i in tqdm(range(len(classifiers))):

        clf = clfs[i]
        clf.optimize_hyperparameters()
        clf.display_general_results()
        clf.display_cv_results()
        print("Test accuracy : {:.03f}".format(clf.get_accuracy(X_test, t_test)))
        print("Test f1-score : {:.03f}".format(clf.get_f1_score(X_test, t_test)))


if __name__ == "__main__":
    main()
