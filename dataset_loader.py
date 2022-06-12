import os

from collections import Counter

import numpy as np
import pandas as pd


# CATEGORICAL_KEYS = {"aids3": ["state", "sex", "T.categ", "zid"],
#                     # "colon_death": ["rx", "sex", "obstruct", "perfor", "adhere", "nodes", "differ", "extent",
#                     #                 "surg", "node4"],
#                     "support2": ["sex", "income", "race", "ca", "dnr", "sfdm2", "dzgroup", "dzclass"]}

# CONTINUOUS_KEYS = {"aids3": ["age"],
#                 #    "colon_death": ["age"],
#                    "support2": ["age", "num.co", "edu", "scoma", "avtisst", "sps", "aps", "surv2m", "surv6m",
#                                 "hday", "diabetes", "dementia", "prg2m", "prg6m", "dnrday", "meanbp", "wblc",
#                                 "hrt", "resp", "temp", "pafi", "alb", "bili", "crea", "sod", "ph", "glucose",
#                                 "bun", "urine", "adlp", "adls", "adlsc"]}


def categories_to_dummies(X, columns):
    """
    Convert categorical variables into a dummy/indicator
    variables for a given list of categorical variables
    in a given dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataset.
    columns : list
        List of columns to convert.

    Returns
    -------
    X_dummy : pandas.DataFrame
        Converted dataset.
    """
    X_dummy = X.copy()
    for col in columns:
        dummies = pd.get_dummies(X_dummy[col], drop_first=True).rename(columns=lambda x: f"{col}_{x}")
        nan_values = -X_dummy[col].isnull() * 1
        nan_values = nan_values.rename(f"{col}_nan")
        dummies = pd.concat([dummies, nan_values], axis=1)
        X_dummy = pd.concat([X_dummy, dummies], axis=1)
        X_dummy = X_dummy.drop([col], axis=1)
    return X_dummy


def continuous_nan(X, columns):
    """
    Add indicator variables of missing values for a
    given list of continuous variables and fill those
    nan values with 0 in a given dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataset.
    columns : list
        List of columns to convert.

    Returns
    -------
    X_with_nan : pandas.DataFrame
        Converted dataset.

    """
    X_with_nan = X.copy()
    for col in columns:
        X = X_with_nan[col].copy()
        X_with_nan = X_with_nan.drop([col], axis=1)
        nan_values = -X.isnull() * 1
        X = X.fillna(value=0)
        nan_values = nan_values.rename(f"{col}_nan")
        X = pd.concat([X, nan_values], axis=1)
        X_with_nan = pd.concat([X, X_with_nan], axis=1)
    return X_with_nan


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
    
    # Load data
    if cfg.DATA.DATASET == "colon_death":
        data = data[data["etype"] == 2]

    # Prepare covariates matrix
    categorical_keys = cfg.DATA.CATEGORICAL_KEYS
    continuous_keys = cfg.DATA.CONTINUOUS_KEYS
    X = data[categorical_keys + continuous_keys]
    X = categories_to_dummies(X, categorical_keys)
    X = continuous_nan(X, continuous_keys)
    col = X.columns.tolist()
    dict_col = {'categorical_keys': categorical_keys, 'continuous_keys': continuous_keys, 'col': col}
    X = X.to_numpy()

    # Prepare targets matrix
    y = data[cfg.DATA.TARGET].to_numpy()
    # Prepare censure matrix
    y_cens = data[cfg.DATA.EVENT].to_numpy().astype("float32")

    print(f"\nDataset {cfg.DATA.DATASET} info:\nNb. col: {len(col)}")
    print(f"Nb unique t: {len(np.unique(y))}\nMin t: {np.min(y)}\nMax t: {np.max(y)}")
    print(f"Nb of event: {Counter(data[cfg.DATA.EVENT])}")

    data = np.concatenate((y.reshape(-1, 1), y_cens.reshape(-1, 1)), axis=1)
    data = np.concatenate((data, X), axis=1)
    data = data.astype("float32")

    return data, dict_col
