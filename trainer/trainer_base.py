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


class TrainerBase(object):

    def __init__(self, cfg, split):
        """
        Trainer base class object.
        """
        self.cfg = cfg

        self.verbose = bool(cfg.VERBOSE)
        self.dataset = cfg.DATA.DATASET
        self.split = split

        self.data_path = cfg.DATA.PATH

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_patience = cfg.TRAIN.PATIENCE

        self.num_epochs = cfg.TRAIN.MAX_EPOCH
        self.model_name = cfg.TRAIN.MODEL
        self.loss_type = cfg.TRAIN.LOSS_TYPE
        self.learning_rate = float(self.cfg.TRAIN.LR)
        self.l2_coeff = float(self.cfg.TRAIN.L2_COEFF)
        self.get_data()

        self.use_cuda = cfg.CUDA
        self.model = None
        # self.X_train_shape = (-1, data.shape[1] - 2)

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
    
    def get_data(self):
        """
        Load the data and extract input (and output in the case of EMD) shapes
        to be use by the model.
        """
        data, self.dict_col = load_data(self.cfg)
        
        self.X_train_shape = (-1, data.shape[1] - 2)
        if self.cfg.TRAIN.MODEL == "emd":
            self.time_shape = int(data[:, 0].max()) + 1
        self.data = data

    def before_train(self, train):
        pass

    def forward(self, batch):
        raise

    def run(self, save_best_model=False):
        train, val, test = self.get_data_random_split()
        if self.cfg.DATA.NO_CENSORED_DATA:
            train = train[train[:, 1] == 1]
        self.before_train(train)
        split_id = f"split{self.split}"

        results = {}

        train_loss_history = []
        val_loss_history = []
        train_cindex_history = []
        val_cindex_history = []

        patience = 0

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_coeff)

        best_c_index = -np.inf

        for epoch in range(self.num_epochs):

            # Train
            concat_pred_train = np.array([]).reshape(0, 3)
            train_epoch_loss = 0
            train_iteration = 0
            self.model.train()
            for batch in iterate_minibatches(train, self.batch_size, shuffle=True):
                concat_pred, loss = self.forward(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                concat_pred_train = np.concatenate((concat_pred_train, concat_pred.data.cpu().numpy()), axis=0)
                train_epoch_loss += loss.data.item()
                train_iteration += 1

            # Record and print result after each epoch
            train_loss = train_epoch_loss / train_iteration
            train_loss_history.append(train_loss)
            train_c_index = concordance_index(concat_pred_train[:, 0], concat_pred_train[:, 2], concat_pred_train[:, 1])
            if train_c_index < 0.5:
                train_c_index = concordance_index(concat_pred_train[:, 0], -concat_pred_train[:, 2], concat_pred_train[:, 1])
            train_cindex_history.append(train_c_index)

            # Val
            concat_pred_val = np.array([]).reshape(0, 3)
            val_epoch_loss = 0
            val_iteration = 0
            self.model.eval()
            for batch in iterate_minibatches(val, self.batch_size):
                concat_pred, loss = self.forward(batch)
                concat_pred_val = np.concatenate((concat_pred_val, concat_pred.data.cpu().numpy()), axis=0)
                val_epoch_loss += loss.data.item()
                val_iteration += 1

            # Record and print result after each epoch
            val_loss = val_epoch_loss / val_iteration
            val_loss_history.append(val_loss)
            val_c_index = concordance_index(concat_pred_val[:, 0], concat_pred_val[:, 2], concat_pred_val[:, 1])
            if val_c_index < 0.5:
                val_c_index = concordance_index(concat_pred_val[:, 0], -concat_pred_val[:, 2], concat_pred_val[:, 1])
            val_cindex_history.append(val_c_index)

            # Plot training and validation curve
            path = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.PARAMS, "Figures/")
            mkdir_p(os.path.dirname(path))
            plot_history(path, f"error_{split_id}", train_loss_history, val_loss_history)
            plot_history(path, f"c_index_{split_id}", train_cindex_history, val_cindex_history)

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


        concat_pred_test = np.array([]).reshape(0, 3)

        test_epoch_loss = 0
        test_iteration = 0
        self.model.eval()
        for batch in iterate_minibatches(test, self.batch_size, shuffle=False):
            concat_pred, loss = self.forward(batch)
            concat_pred_test = np.concatenate((concat_pred_test, concat_pred.data.cpu().numpy()), axis=0)
            test_epoch_loss += loss.data.item()
            test_iteration += 1

        test_loss = test_epoch_loss / test_iteration

        test_c_index = concordance_index(concat_pred_test[:, 0], concat_pred_test[:, 2], concat_pred_test[:, 1])

        if test_c_index < 0.5:
            test_c_index = concordance_index(concat_pred_test[:, 0], -concat_pred_test[:, 2], concat_pred_test[:, 1])

        results['test'] = {'avg_loss': test_loss, 'c_index': test_c_index}
        return results


    def get_data_random_split(self):
        """
        Get data split train/val/test from  5Fold cross-validation.

        Returns
        -------
        train : ndarray
            Training set.
        val : ndarray
            Validation set.
        test : ndarray
            Test set.
        """

        kf = KFold(n_splits=5)

        index_train = []
        index_valid = []
        index_test = []
        for train, test in kf.split(self.data):
            index_train.append(train[:int(len(self.data)*0.6)])
            index_valid.append(train[int(len(self.data)*0.6):])
            index_test.append(test)

        if self.split is not None:
            train = self.data[index_train[self.split]]
            val = self.data[index_valid[self.split]]
            test = self.data[index_test[self.split]]
        else:
            raise NotImplementedError()

        # Normalise the data
        col_name = ["time", "event"] + self.dict_col['col']
        df_train = pd.DataFrame(data=train, columns=col_name)
        df_val = pd.DataFrame(data=val, columns=col_name)
        df_test = pd.DataFrame(data=test, columns=col_name)
        scaler = MinMaxScaler()
        df_train[self.dict_col['continuous_keys']] = scaler.fit_transform(df_train[self.dict_col['continuous_keys']])
        df_val[self.dict_col['continuous_keys']] = scaler.transform(df_val[self.dict_col['continuous_keys']])
        df_test[self.dict_col['continuous_keys']] = scaler.transform(df_test[self.dict_col['continuous_keys']])

        train = df_train.to_numpy()
        val = df_val.to_numpy()
        test = df_test.to_numpy()

        train = self.preprocess_train(train)

        if self.cfg.DATA.ADD_CENS:
            proba = self.cfg.DATA.PROBA
            cens = train[train[:, 1] == 0]
            non_cens = train[train[:, 1] == 1]

            # Add censure cases in the event feature
            p_ = proba - (cens.shape[0] / float(train.shape[0]))
            p_ = (train.shape[0] * p_) / float(non_cens.shape[0])
            ev_new = np.random.binomial(size=non_cens.shape[0], n=1, p=1-p_)
            non_cens[:, 1] = ev_new

            # Modify target for new censured cases
            new_cens = non_cens[non_cens[:, 1] == 0]
            non_cens = non_cens[non_cens[:, 1] == 1]
            tgt_ = new_cens[:, 0]
            g_rand = lambda x: np.random.randint(x)
            new_tgt = list(map(g_rand, tgt_))
            new_cens[:, 0] = new_tgt

            train = np.concatenate((cens, new_cens), axis=0)
            train = np.concatenate((train, non_cens), axis=0)

        return train, val, test


    def preprocess_train(self, train):
        """
        Returns
        -------
        train : ndarray
            Training set.
        val : ndarray
            Validation set.
        test : ndarray
            Test set.
        """
        if self.cfg.DATA.ADD_CENS:
            proba = self.cfg.DATA.PROBA
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

    

