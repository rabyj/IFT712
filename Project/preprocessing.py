import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

class preprocessing():
    
    def importEncode(self):
        """

        Returns:
            [type]: [description]
        """
        
        # import
        test_data = pd.read_csv("data/test.csv")
        train_data = pd.read_csv("data/train.csv")    
        
        # encode labels to be used as the target dataset 
        # encode species to numerical categories
        le = LabelEncoder().fit(train_data.species) 
        t = le.transform(train_data.species)                   

        # the id and species columns are not usefull in our analysis
        train = train_data.drop(['species', 'id'], axis=1)  
        test = test_data.drop(['id'], axis=1)

        # center data around 0
        scaled_train = train.copy()
        scaled_test = test.copy()
        col_names = scaled_train.columns
        features_train = scaled_train[col_names]
        features_test = scaled_test[col_names]
        # use StandardScaler
        scaler = StandardScaler().fit(features_train.values)
        features_train = scaler.transform(features_train.values)
        features_test = scaler.transform(features_test.values)
        # create dfs from scaled data
        scaled_train[col_names] = features_train
        scaled_test[col_names] = features_test
        
        return scaled_train, scaled_test, t


    def trainTestSplit(self, train, t):
        """

        Args:
            train ([type]): [description]
            t ([type]): [description]

        Returns:
            [type]: [description]
        """
        X_train, X_test, t_train, t_test = None, None, None, None

        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        for train_index, test_index in kfold.split(train, t):
            X_train, X_test = train.values[train_index], train.values[test_index]
            t_train, t_test = t[train_index], t[test_index]
        
        return X_train, X_test, t_train, t_test