import os
import numpy as np
import pandas as pd
import pdb

# def categories_to_dummies(X, columns):
#     """
#     Convert categorical variables into a dummy/indicator
#     variables for a given list of categorical variables
#     in a given dataset.

#     Parameters
#     ----------
#     X : pandas.DataFrame
#         Dataset.
#     columns : list
#         List of columns to convert.

#     Returns
#     -------
#     X_dummy : pandas.DataFrame
#         Converted dataset.
#     """
#     X_dummy = X.copy()
#     for col in columns:
#         dummies = pd.get_dummies(X_dummy[col]).rename(columns=lambda x: f"{col}_{x}")
#         if X_dummy[col].isnull().sum() > 0:
#             nan_values = X_dummy[col].isnull() * 1
#             nan_values = nan_values.rename(f"{col}_nan")
#             dummies = pd.concat([dummies, nan_values], axis=1)
#         X_dummy = pd.concat([X_dummy, dummies], axis=1)
#         X_dummy = X_dummy.drop([col], axis=1)
#     return X_dummy


# def continuous_nan(X, columns):
#     """
#     Add indicator variables of missing values for a
#     given list of continuous variables and fill those
#     nan values with 0 in a given dataset.

#     Parameters
#     ----------
#     X : pandas.DataFrame
#         Dataset.
#     columns : list
#         List of columns to convert.

#     Returns
#     -------
#     X_with_nan : pandas.DataFrame
#         Converted dataset.

#     """
#     X_with_nan = X.copy()
#     for col in columns:
#         if X_with_nan[col].isnull().sum() > 0:
#             X = X_with_nan[col].copy()
#             X_with_nan = X_with_nan.drop([col], axis=1)
#             nan_values = X.isnull() * 1
#             X = X.fillna(value=0)
#             nan_values = nan_values.rename(f"{col}_nan")
#             X = pd.concat([X, nan_values], axis=1)
#             X_with_nan = pd.concat([X, X_with_nan], axis=1)
#     return X_with_nan


# def transform(data, cfg):
#     """
#     Returns
#     -------
#     data : ndarray
#         Dataset.
#     dict_col : dict
#         Dictionary containing the list of continuous ('continuous_keys'),
#         categorical ('categorical_keys') and all features ('col').
#     """
#     # Prepare covariates matrix
#     categorical_keys = cfg.DATA.CATEGORICAL_KEYS
#     continuous_keys = cfg.DATA.CONTINUOUS_KEYS
#     # if cfg.DATA.NORMALIZE:
#     #     for k in continuous_keys:
#     #         idxs = data[data[k].notnull()].index
#     #         data.loc[idxs, k] = ((data.loc[idxs, k] - data.loc[idxs, k].mean(axis=0)).values
#     #                             / (data.loc[idxs, k].std(axis=0) + 1e-6))
#     X = data[categorical_keys + continuous_keys]
#     dict_col = {'categorical_keys': cfg.DATA.CATEGORICAL_KEYS,
#                 'continuous_keys': cfg.DATA.CONTINUOUS_KEYS,
#                 'col': data.columns.tolist()}
#     X = categories_to_dummies(X, categorical_keys)
#     X = continuous_nan(X, continuous_keys)
#     X = X.to_numpy()

#     y = data[cfg.DATA.TARGET].to_numpy().astype("float32")
#     y_cens = data[cfg.DATA.EVENT].to_numpy().astype("float32")

#     data = np.concatenate((y.reshape(-1, 1), y_cens.reshape(-1, 1)), axis=1)
#     data = np.concatenate((data, X), axis=1)
#     data = data.astype("float32")  
#     return data, dict_col


def load_data(cfg):
    """
    General function to load the datasets.

    Included datasets: Aids 3, COLON dataset and SUPPORT2.

    For more information about those datasets check the related
    `.md` files in this folder.

    Returns
    -------
    data : ndarray
        Dataset.
    dict_col : dict
        Dictionary containing the list of continuous ('continuous_keys'),
        categorical ('categorical_keys') and all features ('col').
    """
    path = os.path.join(cfg.DATA.PATH, f"{cfg.DATA.DATASET}.csv")
    data = pd.read_csv(path, delimiter=',', header=0, low_memory=False)
    data = transform(data, cfg)
    return data
