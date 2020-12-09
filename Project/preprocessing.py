"""Data preprocessing steps. Center and scale data. Apply PCA. Split for cross-validation.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocessor():
    """Contains pre-processing and data transformation steps.

    Attributes:
        last_data : Last panda dataframe created from raw csv input
        label_encoder : sklearn object used to encode target labels into number
        scaler : sklearn object used to center and reduce data
        pca : sklearn object used to transform data using PCA
    """
    def __init__(self):
        self.last_data = None
        self.label_encoder = None
        self.scaler = None
        self.pca = None


    def import_data(self, path):
        """Read CSV data into Panda dataframe.

        Args:
            path (string) : path of csv

        Sets attributes:
            self.last_data
        """
        self.last_data = pd.read_csv(path)


    def encode_labels(self, use_new_encoder=False):
        """Encode last loaded dataframe class labels into numerical classes.

        Args:
            use_new_encoder (bool) : If true, create new label encoder from data
                                     and set it as instance attribute "label_encoder".
        Returns:
            X_train (pd dframe) : training data (with no "id"/"species" original columns.)
            t_train (pd dframe) : array-like target labels as integers
        """
        if not use_new_encoder and self.label_encoder is None:
            print("No label encoder created yet. Using data to create it.")
            use_new_encoder = True

        if use_new_encoder:
            self.label_encoder = LabelEncoder().fit(self.last_data.species)

        t_train = self.label_encoder.transform(self.last_data.species)

        # the id and species columns are not useful in the training once encoded
        X_train = self.last_data.drop(["species", "id"], axis=1)

        return X_train, t_train


    def scale_data(self, data_df, use_new_scaler=False):
        """Standardize features by removing the (training) mean and scaling to unit variance.

        Args:
            use_new_scaler (bool) : If true, compute new data scaler from data
                                     and set it as instance attribute "scaler".
        Returns:
            Scaled features (pd dframe)
        """
        if not use_new_scaler and self.scaler is None:
            print("No scaler created yet. Using data to create it.")
            use_new_scaler = True

        if use_new_scaler:
            self.scaler = StandardScaler().fit(data_df.values)

        scaled_data = self.scaler.transform(data_df.values)
        scaled_df = pd.DataFrame(scaled_data, index=data_df.index, columns=data_df.columns)

        return scaled_df


    def apply_pca(self, data, n_components=10, use_new_pca=False):
        """Apply PCA on the data.

        Args:
            data (pd dframe) : Data to transform.
            n_components : Number of PCA components to keep if computing a new PCA.
            use_new_pca (bool) : If true, compute new PCA with n_components from data
                                     and set it as instance attribute "pca".
        Returns:
            Transformed features (pd dframe)
        """
        if not use_new_pca and self.pca is None:
            print("No PCA tranformation computed yet. Using data to create it.")
            use_new_pca = True

        if use_new_pca:
            pca = PCA(n_components=n_components, whiten=True)
            self.pca = pca.fit(data)

        return pd.DataFrame(self.pca.transform(data))

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
