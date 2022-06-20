import os

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


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
    # Load data
    path = os.path.join(cfg.DATA.PATH, f"{cfg.DATA.DATASET}.csv")
    data = pd.read_csv(path, delimiter=',', header=0, low_memory=False)

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
    y = data[cfg.DATA.TARGET].to_numpy() / 365 # (HARDCODED FOR NOW) Convert to years
    # Prepare censure matrix
    y_cens = data[cfg.DATA.EVENT].to_numpy().astype("float32")

    print(f"\nDataset {cfg.DATA.DATASET} info:\nNb. col: {len(col)}")
    print(f"Nb unique t: {len(np.unique(y))}\nMin t: {np.min(y)}\nMax t: {np.max(y)}")
    print(f"Nb of event: {Counter(data[cfg.DATA.EVENT])}")

    data = np.concatenate((y.reshape(-1, 1), y_cens.reshape(-1, 1)), axis=1)
    data = np.concatenate((data, X), axis=1)
    data = data.astype("float32")

    return data, dict_col

def build_kfold_splits(cfg, data, dict_col, k=5):
    """
    Generate random data splits of train/val/test using KFold cross validation.

    Returns
    -------
    train : ndarray
        Training set.
    val : ndarray
        Validation set.
    test : ndarray
        Test set.
    """
    kf = KFold(n_splits=k)
    time_shape = int(data[:, 0].max()) + 1
    for train_val_idxs, test_idxs in kf.split(data):
        train_idxs = train_val_idxs[:int(len(train_val_idxs)*0.65)]
        val_idxs = train_val_idxs[int(len(train_val_idxs)*0.65):]

        columns = ["time", "event"] + dict_col['col']
        df_train = pd.DataFrame(data=data[train_idxs], columns=columns)
        df_val = pd.DataFrame(data=data[val_idxs], columns=columns)
        df_test = pd.DataFrame(data=data[test_idxs], columns=columns)

        if cfg.DATA.NORMALIZE:
            scaler = MinMaxScaler()
            cont_cols = dict_col['continuous_keys']
            df_train[cont_cols] = scaler.fit_transform(df_train[cont_cols])
            df_val[cont_cols] = scaler.transform(df_val[cont_cols])
            df_test[cont_cols] = scaler.transform(df_test[cont_cols])

        train = df_train.to_numpy()
        val = df_val.to_numpy()
        test = df_test.to_numpy()

        if cfg.DATA.ADD_CENS:
            train = add_cens_to_train(cfg.DATA.PROBA, train)

        yield train, val, test


def add_cens_to_train(proba, train):
    """
    Returns
    -------
    train : ndarray
        Training set.
    """
    cens = train[train[:, 1] == 0]
    non_cens = train[train[:, 1] == 1]

    # Add censure cases in the event feature
    p_ = proba - (cens.shape[0] / float(train.shape[0]))
    p_ = (train.shape[0] * p_) / float(non_cens.shape[0])
    non_cens[:, 1] = np.random.binomial(size=non_cens.shape[0], n=1, p=1-p_)

    # Modify target for new censured cases
    new_cens = non_cens[non_cens[:, 1] == 0]
    non_cens = non_cens[non_cens[:, 1] == 1]
    tgt_ = new_cens[:, 0]
    new_tgt = list(map(lambda x: np.random.randint(x), tgt_))
    new_cens[:, 0] = new_tgt

    train = np.concatenate((cens, new_cens), axis=0)
    train = np.concatenate((train, non_cens), axis=0)

    return train