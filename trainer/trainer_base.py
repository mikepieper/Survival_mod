import os
import pprint

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from dataset_loader import load_data
from utils.utils import mkdir_p, iterate_minibatches, save_model
from visualization import plot_history

import pdb

class TrainerBase(object):

    def __init__(self, cfg):
        """
        Trainer base class object.
        """
        self.cfg = cfg
        self.verbose = bool(cfg.VERBOSE)
        self.dataset = cfg.DATA.DATASET
        self.data_path = cfg.DATA.PATH
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_patience = cfg.TRAIN.PATIENCE
        self.num_epochs = cfg.TRAIN.MAX_EPOCH
        self.model_name = cfg.TRAIN.MODEL
        self.loss_type = cfg.TRAIN.LOSS_TYPE
        self.learning_rate = float(self.cfg.TRAIN.LR)
        self.l2_coeff = float(self.cfg.TRAIN.L2_COEFF)
        self.use_cuda = cfg.CUDA
        self.X_train_shape = cfg.train_shape
        self.model = None

    def process_batch(self, data):
        """
        Process batch of data to extract the time, the event
        and the explanatory variables matrix and, to compute
        the lower triangle and diagonal by block matrix.

        Parameters
        ----------
        data : ndarray
            Data to process.

        Returns
        -------
        time : ndarray
            Time.
        event : ndarray
            Event.
        X : ndarray
            Explanatory variables matrix.
        tril : ndarray
            Lower triangular matrix.
        tied_matrix : ndarray
            Diagonal by block matrix.
        """
        data = data[np.argsort(data[:,0]),:] # Ascending sort by time
        time, event, X = np.hsplit(data, [1, 2])
        
        time = torch.from_numpy(time)
        event = torch.from_numpy(event)
        if self.cfg.DATA.DEATH_AT_CENSOR_TIME:
            event = torch.ones(event.size())
        
        X = torch.from_numpy(X)


        if self.use_cuda:
            time = time.cuda()
            event = event.cuda()
            X = X.cuda()

        return time, event, X


    def before_train(self, train):
        pass

    def forward(self, batch):
        raise

    def run(self, train, val, test, split, save_best_model=False):
        # Extract input (and output in the case of EMD) shapes
        # to be use by the model.
        self.X_train_shape = (-1, train.shape[1] - 2)
        if self.cfg.DATA.NO_CENSORED_DATA:
            train = train[train[:, 1] == 1]
        self.before_train(train)

        results = {}

        train_loss_history = []
        val_loss_history = []
        train_cindex_history = []
        val_cindex_history = []

        patience = 0

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_coeff)

        best_c_index = -np.inf
        
        flip = False
        for epoch in range(self.num_epochs):

            # Train
            self.model.train()
            train_epoch_loss = 0
            train_iteration = 0
            preds = np.array([]).reshape(0, 3)
            for batch in iterate_minibatches(train, self.batch_size, shuffle=True):
                concat_pred, loss = self.forward(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = np.concatenate((preds, concat_pred.data.cpu().numpy()), axis=0)
                train_epoch_loss += loss.data.item()
                train_iteration += 1

            # Record and print result after each epoch
            train_loss = train_epoch_loss / train_iteration
            train_loss_history.append(train_loss)
            train_c_index = concordance_index(preds[:, 0], preds[:, 2], preds[:, 1])
            flip = False if train_c_index >= 0.5 else True
            if flip:
                preds[:, 2] = -preds[:, 2]
            train_c_index = concordance_index(preds[:, 0], preds[:, 2], preds[:, 1])
            train_cindex_history.append(train_c_index)

            # Val
            self.model.eval()
            val_epoch_loss = 0
            val_iteration = 0
            preds = np.array([]).reshape(0, 3)
            for batch in iterate_minibatches(val, self.batch_size):
                concat_pred, loss = self.forward(batch)
                preds = np.concatenate((preds, concat_pred.data.cpu().numpy()), axis=0)
                val_epoch_loss += loss.data.item()
                val_iteration += 1

            # Record and print result after each epoch
            val_loss = val_epoch_loss / val_iteration
            val_loss_history.append(val_loss)
            if flip:
                preds[:, 2] = -preds[:, 2]
            val_c_index = concordance_index(preds[:, 0], preds[:, 2], preds[:, 1])
            val_cindex_history.append(val_c_index)

            # Plot training and validation curve
            path = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.PARAMS, "Figures/")
            mkdir_p(os.path.dirname(path))
            plot_history(path, f"error_split_{split}", train_loss_history, val_loss_history)
            plot_history(path, f"c_index_split_{split}", train_cindex_history, val_cindex_history)

            if val_c_index > best_c_index:
                best_c_index = val_c_index
                results['train'] = {'avg_loss': train_loss, 'c_index': train_c_index}
                results['val'] = {'avg_loss': val_loss, 'c_index': val_c_index, 'best_epoch': epoch}
                if save_best_model:
                    save_model(self.model, os.path.join(self.cfg.OUTPUT_DIR, self.cfg.PARAMS, "Models"), "best.pth")
                patience = 0
            else:
                patience += 1

            if patience > self.max_patience:
                break


        self.model.eval()
        test_epoch_loss = 0
        test_iteration = 0
        preds = np.array([]).reshape(0, 3)
        for batch in iterate_minibatches(test, self.batch_size, shuffle=False):
            concat_pred, loss = self.forward(batch)
            preds = np.concatenate((preds, concat_pred.data.cpu().numpy()), axis=0)
            test_epoch_loss += loss.data.item()
            test_iteration += 1

        test_loss = test_epoch_loss / test_iteration
        if flip:
            preds[:, 2] = -preds[:, 2]
        test_c_index = concordance_index(preds[:, 0], preds[:, 2], preds[:, 1])

        results['test'] = {'avg_loss': test_loss, 'c_index': test_c_index}
        return results

    
    # def build_kfold_splits(self):
    #     """
    #     Generate random data splits of train/val/test using KFold cross validation.

    #     Returns
    #     -------
    #     train : ndarray
    #         Training set.
    #     val : ndarray
    #         Validation set.
    #     test : ndarray
    #         Test set.
    #     """
    #     kf = KFold(n_splits=5)

    #     self.index_train = []
    #     self.index_valid = []
    #     self.index_test = []
    #     for train, test in kf.split(self.data):
    #         self.index_train.append(train[:int(len(self.data)*0.65)])
    #         self.index_valid.append(train[int(len(self.data)*0.65):])
    #         self.index_test.append(test)


    # def get_data_random_split(self, split):
    #     columns = ["time", "event"] + self.dict_col['col']
    #     df_train = pd.DataFrame(data=self.data[self.index_train[self.split]], columns=columns)
    #     df_val = pd.DataFrame(data=self.data[self.index_valid[self.split]], columns=columns)
    #     df_test = pd.DataFrame(data=self.data[self.index_test[self.split]], columns=columns)

    #     if self.cfg.DATA.NORMALIZE:
    #         scaler = MinMaxScaler()
    #         cont_cols = self.dict_col['continuous_keys']
    #         df_train[cont_cols] = scaler.fit_transform(df_train[cont_cols])
    #         df_val[cont_cols] = scaler.transform(df_val[cont_cols])
    #         df_test[cont_cols] = scaler.transform(df_test[cont_cols])

    #     train = df_train.to_numpy()
    #     val = df_val.to_numpy()
    #     test = df_test.to_numpy()

    #     if self.cfg.DATA.ADD_CENS:
    #         train = self.add_cens_to_train(train)

    #     return train, val, test


    # def add_cens_to_train(self, train):
    #     """
    #     Returns
    #     -------
    #     train : ndarray
    #         Training set.
    #     """
    #     proba = self.cfg.DATA.PROBA
    #     cens = train[train[:, 1] == 0]
    #     non_cens = train[train[:, 1] == 1]

    #     # Add censure cases in the event feature
    #     p_ = proba - (cens.shape[0] / float(train.shape[0]))
    #     p_ = (train.shape[0] * p_) / float(non_cens.shape[0])
    #     non_cens[:, 1] = np.random.binomial(size=non_cens.shape[0], n=1, p=1-p_)

    #     # Modify target for new censured cases
    #     new_cens = non_cens[non_cens[:, 1] == 0]
    #     non_cens = non_cens[non_cens[:, 1] == 1]
    #     tgt_ = new_cens[:, 0]
    #     new_tgt = list(map(lambda x: np.random.randint(x), tgt_))
    #     new_cens[:, 0] = new_tgt

    #     train = np.concatenate((cens, new_cens), axis=0)
    #     train = np.concatenate((train, non_cens), axis=0)

    #     return train

    

