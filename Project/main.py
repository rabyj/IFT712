import warnings
warnings.filterwarnings('ignore')

# preprocessing
from preprocessing import preprocessing

# models
from LR_clf import LR_clf
from Perceptron_clf import Perceptron_clf
from SVM_clf import SVM_clf
from MLP_clf import MLP_clf
from RF_clf import RF_clf
from NB_clf import NB_clf

from tqdm import tqdm

if __name__ == '__main__':
    
    
    preprocessing = preprocessing()
    train, test, t = preprocessing.importEncode()
    X_train, X_test, t_train, t_test = preprocessing.trainTestSplit(train, t)
    
    classifiers = [LR_clf, Perceptron_clf, SVM_clf, MLP_clf, RF_clf, NB_clf]
    clfs = []
    
    for clf in classifiers:
        clfs.append(clf(X_train, X_test, t_train, t_test))
    
    
    for i in tqdm(range(len(classifiers))):
        
        clfs[i].getHyperParameters()
        clfs[i].trainDataset()
        clfs[i].displayResults()
    
    
    
