import argparse
import copy
import datetime
import itertools
import os
import pprint
import random
import sys

import dateutil
import dateutil.tz
import numpy as np
import torch
from sklearn.model_selection import KFold


from shutil import copyfile

from trainer import get_algo
from utils.config import cfg, cfg_from_file
from utils.utils import create_output_dir, mkdir_p


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    """
    Parser for the command line arguments.

    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Launch survival experiment.")
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help="Optional config file.",
                        default="aids3/cox_aids3.yml", type=str)
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        type=str,
                        default='0')
    parser.add_argument('--manual_seed',
                        type=int,
                        help="manual seed",
                        default=1234)

    args = parser.parse_args()
    return args


def set_cfg_params(cfg, params):
    cfg.PARAMS = f"LR{params[0]}_L2{params[1]}"
    cfg.TRAIN.LR = params[0]
    cfg.TRAIN.L2_COEFF = params[1]

    if cfg.TRAIN.MODEL == "emd":
        cfg.PARAMS += f"_PRIOR{params[2]}"
        cfg.EMD.PRIOR = params[2]
    return cfg


def create_grid_search(cfg):
    param_lr = [float(i) for i in cfg.TRAIN.LR]
    param_l2_coeff = [float(i) for i in cfg.TRAIN.L2_COEFF]
    if cfg.TRAIN.MODEL == "emd":
        param_prior = [float(i) for i in cfg.EMD.PRIOR]
        params = list(itertools.product(param_lr, param_l2_coeff, param_prior))
    else:
        params = list(itertools.product(param_lr, param_l2_coeff))
    return params


def run_cx_val(cfg):
    """
        Mike's Notes: I swapped the loops of params and split.
        The previous approach took the best cindex across all params for each split.
        This is backwards. It should be: For each param combo. Take the test cindex 
        across all splits.
    """
    # kf = KFold(n_splits=5)
    params_grid = create_grid_search(cfg)

    test_cindices = []
    splits = eval(cfg.DATA.SPLITS) or [0, 1, 2, 3, 4]
    best_val_cindex, best_test_cindex, best_params = 0, 0, None
    for split in splits:
    # for params in params_grid:

        print(f"\n\n\nSplit: {split}")
        opt_val_cindices = []
        opt_test_cindices = []
        for params in params_grid:
        # for split in splits:
            cfg = set_cfg_params(cfg, params)
            print(f"\nParam: {params}\n")

            results = get_algo(cfg, split).run()

            val_cindex = results['val']['c_index']
            test_cindex = results['test']['c_index']

            opt_val_cindices.append(val_cindex)
            opt_test_cindices.append(test_cindex)

        test_cindex = opt_test_cindices[np.argmax(opt_val_cindices)]
        test_cindices.append(test_cindex)

    print(f"\n{test_cindices}")
    print(f"\nTest cindex mean, std: {np.mean(test_cindices).round(3)}, {np.std(test_cindices).round(3)}")

if __name__ == "__main__":
    # Loading arguments
    args = parse_args()
    args.cfg_file = os.path.join("config", args.cfg_file)
    cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id

    # Setting seeds for reproducibility
    print(f"\nPyTorch/Random seed: {args.manual_seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manual_seed)

    create_output_dir(cfg, args.cfg_file)
    run_cx_val(cfg)

    
