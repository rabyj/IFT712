"""Data preprocessing steps. Center and scale data. Apply PCA. Split for cross-validation.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocessor():
    """Contains pre-processing and data transformation steps.
    Minimal usage steps:
        - import
        - encode labels
        - split into train/test

    Attributes:
        last_data : Last panda dataframe created from raw csv input
        label_encoder : sklearn object used to encode target labels into number
        scaler : sklearn object used to center and reduce data
        pca : sklearn object used to transform data using PCA
    """
    def __init__(self):
        """Initialize all attributes to None."""
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
            X (pd dframe) : training data (with no "id"/"species" original columns.)
            t (np array) : target labels as integers
        """
        if not use_new_encoder and self.label_encoder is None:
            print("No label encoder created yet. Using data to create it.")
            use_new_encoder = True

        if use_new_encoder:
            self.label_encoder = LabelEncoder().fit(self.last_data.species)

        t = self.label_encoder.transform(self.last_data.species)

        # the id and species columns are not useful in the training once encoded
        X = self.last_data.drop(["species", "id"], axis=1)

        return X, t


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


    def apply_pca(self, data, n_components=0.8, use_new_pca=False):
        """Apply PCA on the data.

        Args:
            data (pd dframe) : Data to transform.
            n_components (int or float):
                If int, number of PCA components to keep if computing a new PCA.
                If float between 0 and 1, components needed to explain percentage of variance specified.
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

    def train_test_split(self, data, t, n_split=5):
        """Split the available data using StratifiedKFold.
        Uses the same shuffling random state each time.

        Args:
            data (pd dframe) : Features
            t (np array) : Targets
            n_split (int) : 1/(test ratio). By default, value=5 gives 80%/20% train/test.

        Returns:
            X_train (pd dframe)
            X_test (pd dframe)
            t_train (np array)
            t_test (np array)
        """
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

        train_slice, test_slice = next(kfold.split(data, t))
        X_train, X_test = data.iloc[train_slice], data.iloc[test_slice]
        t_train, t_test = t[train_slice], t[test_slice]

        return X_train, X_test, t_train, t_test
