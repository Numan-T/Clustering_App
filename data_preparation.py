from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


# -----------------------------------------------Import Data:
# Specify csv filename and settings here
csv_filename = "Daten_Numan_verschickt.csv"
csv_encoding = "utf8"
csv_delimiter = ";"
csv_decimal = ","
csv_missing_values = 99999  # how are missing values represented in the csv file
df = pd.read_csv(csv_filename,
                 delimiter=csv_delimiter,
                 decimal=csv_decimal,
                 encoding=csv_encoding)
df = df.replace(csv_missing_values, np.nan)

features = list(df.columns)
default_features = ["CMJ_Diff", "DJ_RSI_Diff", "BPr_Diff",
                    "D_Verl_Diff", "YB_Diff", "MB_Diff", "HK_Diff"]


# -----------------------------------------------

def drop_sparse_rows(X, num_allowed_sparse_features=0.5):
    """
    Drop rows with missing values from the input data X. The parameter num_allowed_sparse_features defines how many empty values in a row are allowed. It can be a percentage or the absolut number.
    """
    n_features = X.shape[1]
    if 0 < num_allowed_sparse_features < 1:  # percentage entered
        num_allowed_sparse_features = int(
            num_allowed_sparse_features * n_features)
    adjusted_X = X.dropna(thresh=n_features-num_allowed_sparse_features)
    return adjusted_X


def impute_X(X):
    """ 
    Replace NaN values (empty cells) of the data X with the median value of the column/feature.
    """
    imputer = SimpleImputer(strategy='median')
    imputed_X = imputer.fit_transform(X)
    return imputed_X
