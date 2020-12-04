"""Data preprocessing steps. Center and scale data. Apply PCA. Split for cross-validation.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocessor():

    def import_encode(self):
        """

        Returns:
           scaled_train [np.array] : data
           t_train [np.array] : labels
        """

        # import
        train_data = pd.read_csv("data/train.csv")

        # encode labels to be used as the target
        # encode species to numerical categories
        le = LabelEncoder().fit(train_data.species)
        t_train = le.transform(train_data.species)

        # the id and species columns are not useful in our analysis once encoded
        train = train_data.drop(["species", "id"], axis=1)

        # center data around 0
        scaled_train = train.copy()
        col_names = scaled_train.columns
        features_train = scaled_train[col_names]

        # use StandardScaler
        scaler = StandardScaler().fit(features_train.values)
        features_train = scaler.transform(features_train.values)

        # create dfs from scaled data
        scaled_train[col_names] = features_train

        #PCA
        pca = PCA(n_components=10, whiten=True)
        pca.fit(scaled_train)
        scaled_train = pd.DataFrame(pca.transform(scaled_train))

        return scaled_train, t_train


    # def train_valid_split(self, train, t):
    #     """

    #     Args:
    #         train (np.array):
    #         t (np.array):

    #     Returns:
    #         X_train [np.array]
    #         X_valid [np.array]
    #         t_train [np.array]
    #         t_valid [np.array]
    #     """
    #     X_train, X_valid, t_train, t_valid = None, None, None, None

    #     #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    #     kfold = StratifiedKFold(n_splits=5, shuffle=True)

    #     for train_index, test_index in kfold.split(train, t):
    #         X_train, X_valid = train.values[train_index], train.values[test_index]
    #         t_train, t_valid = t[train_index], t[test_index]

    #     return X_train, X_valid, t_train, t_valid
