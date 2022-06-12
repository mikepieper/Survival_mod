import os
import datetime
import dateutil.tz
from shutil import copyfile
import errno

import numpy as np
import torch


def mkdir_p(path):
    """
    Make a directory.

    Parameters
    ----------
    path : str
        Path to the directory to make.

    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def tgt_leq_tgt(time):
    """
    Lower triangular matrix where A_ij = 1 if t_i leq t_j.

    Parameters
    ----------
    time: ndarray
        Time sorted in ascending order.

    Returns
    -------
    tril: ndarray
        Lower triangular matrix.
    """
    time = time.astype(np.float32)
    t_i = time.reshape(1, -1)
    t_j = time.reshape(-1, 1)
    tril = np.where(t_i <= t_j, 1., 0.)
    tril = tril.astype(np.float32)
    return tril


def tgt_equal_tgt(time):
    """
    Used for tied times. Returns a diagonal by block matrix.
    Diagonal blocks of 1 if same time.
    Sorted over time. A_ij = i if t_i == t_j.

    Parameters
    ----------
    time: ndarray
        Time sorted in ascending order.

    Returns
    -------
    tied_matrix: ndarray
        Diagonal by block matrix.
    """
    time = time.astype(np.float32)
    t_i = time.reshape(1, -1)
    t_j = time.reshape(-1, 1)
    tied_matrix = np.where(t_i == t_j, 1., 0.)
    tied_matrix = tied_matrix.astype(np.float32)

    assert(tied_matrix.ndim == 2)
    block_sizes = np.sum(tied_matrix, axis=1)
    block_index = np.sum(tied_matrix - np.triu(tied_matrix), axis=1)

    tied_matrix = tied_matrix * (block_index / block_sizes)[:, np.newaxis]
    return tied_matrix


def iterate_minibatches(data, batchsize=32, shuffle=False):
    """
    Iterate minibatches.

    Parameters
    ----------
    data : ndarray
        Dataset to iterate over.
    batchsize : int
        Batch size. Default: 32.
    shuffle : bool
        Whether to shuffle the data before iterating over ot not.
        Default: False.

    Returns
    -------
    ndarray
        Yield minibatches.
    """
    if shuffle:
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]

    for start_idx in range(0, data.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield data[excerpt]

    if start_idx + batchsize != data.shape[0]:
        excerpt = slice(start_idx + batchsize, data.shape[0])
        yield data[excerpt]


def create_output_dir(cfg, cfg_file):
    # Create Timestamp
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    cfg.TIMESTAMP = timestamp
    # Create Output Dir
    output_dir = "results/%s_%s_%s" % (cfg.DATA.DATASET, cfg.CONFIG_NAME, timestamp)
    cfg.OUTPUT_DIR = output_dir
    mkdir_p(output_dir)
    # Save config file
    copyfile(cfg_file, os.path.join(output_dir, "config.yml"))


def save_model(model, save_dir, filename):
    """
    Save the model.

    Parameters
    ----------
    message : srt
        Message to include in the path of the model to save.
    """
    mkdir_p(os.path.dirname(save_dir))
    torch.save(model.state_dict(), os.path.join(save_dir, filename))


def load_model(model, filepath):
    """
    Load best model for a given split id.

    Parameters
    ----------
    split_id : str
        Split id under the form "split{split#}".
    """
    state_dict = torch.load(filepath, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    print(f"\nLoad from: {filepath}")