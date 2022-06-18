import math

import numpy as np
import torch
import torch.nn.functional as F

from lifelines import KaplanMeierFitter

from models import MLP
from trainer.trainer_base import TrainerBase
from utils.loss import emd_loss


class EMDTrainer(TrainerBase):
    def __init__(self, cfg):
        """
        EMD trainer.
        """
        super(EMDTrainer, self).__init__(cfg)

        if self.loss_type == "emd_loss":
            self.emd_loss = emd_loss
        else:
            raise NotImplementedError()

        self.div_time = int(cfg.TRAIN.DIV_TIME)
        self.time_shape = cfg.time_shape
        self.dtime_shape = math.ceil(self.time_shape / self.div_time)
       
        self.model = MLP(cfg, input_shape=self.X_train_shape, output_shape=self.dtime_shape + 1)
        if self.use_cuda:
            self.model = self.model.cuda()


    def before_train(self, train_data):
        """
        Get distance matrix for EMD.

        Parameters
        ----------
        train_data : ndarray
            Training set.
        """
        self.build_survival_estimate(train_data)
        self.get_distance_matrix(train_data)


    def get_distance_matrix(self, train_data):
        """
        Compute the distance matrix for EMD.

        Parameters
        ----------
        train_data : ndarray
            Training set.
        """
        non_c_event_times = train_data[:, 0][train_data[:, 1] == 1]
        hist = np.histogram(non_c_event_times.tolist(), bins=list(range(0, math.ceil(self.time_shape / self.div_time))))
        hist = hist[0]

        prior = float(self.cfg.EMD.PRIOR)
        elts = (prior + hist).tolist()
        self.distance_mat = np.array([elt * (len(elts)) / (sum(elts)) for elt in elts])
        self.distance_mat = torch.from_numpy(self.distance_mat)
        self.distance_mat = self.distance_mat.float()
        padding = torch.zeros(2)
        self.distance_mat = torch.cat([self.distance_mat, padding])

        if self.use_cuda:
            self.distance_mat = self.distance_mat.cuda()


    def forward(self, batch):
        """
        Compute the model loss and prediction.

        Parameters
        ----------
        batch : ndarray
            Batch data.

        Returns
        -------
        concat_pred : ndarray
            Matrix containing the model predictions.
        loss : float
            Model loss.
        """
        time, event, X = self.process_batch(batch)
        self.build_survival_estimate(batch)

        survival_estimate = self.survival_estimate[time.cpu().numpy().astype("int32")].astype("float32")

        if survival_estimate.shape[0] != X.shape[0]:
            survival_estimate = survival_estimate[:X.shape[0], :]

        if self.div_time != 1:
            step_time = [i for i in range(1, survival_estimate.shape[1], self.div_time)]

            surv = []
            for stp in range(len(step_time)):
                if stp < len(step_time) - 1:
                    surv.append(np.median(survival_estimate[:, step_time[stp]:step_time[stp + 1]], axis=1))
                else:
                    surv.append(np.median(survival_estimate[:, step_time[stp]:], axis=1))

            survival_estimate = np.array(surv).T

        survival_estimate = torch.from_numpy(survival_estimate)

        if self.use_cuda:
            survival_estimate = survival_estimate.cuda()

        time_output = self.model(X)
        time_output = F.softmax(time_output, 1)

        # Prepare target
        time_step = torch.arange(0, self.dtime_shape).unsqueeze(0).repeat(X.size()[0], 1)
        if self.use_cuda:
            time_step = time_step.cuda()

        mat_time = time.unsqueeze(0).repeat(self.dtime_shape, 1).transpose(0, 1)
        time_step = time_step - mat_time
        time_step = time_step >= 0
        time_step = time_step.float()

        time_step_cens = time_step * (1. - survival_estimate)
        event_cases = event.unsqueeze(1).repeat(1, self.dtime_shape)
        time_step = time_step * event_cases
        time_step_cens = time_step_cens * (event_cases == 0).float()
        cdf_time = time_step + time_step_cens

        ones = torch.ones(X.size()[0], 1)
        if self.use_cuda:
            ones = ones.cuda()

        cdf_time_ = torch.cat((cdf_time, ones), 1)
        cdf_pred_ = torch.cumsum(time_output, 1)
        loss = self.emd_loss(cdf_pred_, cdf_time_, self.distance_mat)

        rank_output = -torch.mm(cdf_pred_, self.distance_mat.view(-1, 1))

        concat_pred = torch.cat((time.unsqueeze(-1), event.unsqueeze(-1)), 1)
        concat_pred = torch.cat((concat_pred, rank_output), 1)

        return concat_pred, loss


    def get_data_random_split(self):
        train, val, test = super().get_data_random_split()
        self.build_survival_estimate(train)
        return train, val, test

    def build_survival_estimate(self, train):
        kmf = KaplanMeierFitter()
        kmf.fit(train[:, 0], event_observed=train[:, 1])
        timeline = np.array(kmf.survival_function_.index)
        KM_estimate = kmf.survival_function_.values.flatten()
        
        survival = []
        for i in range(self.time_shape):
            surv = np.zeros(self.time_shape)
            prob = 1.
            for j in range(self.time_shape):
                if j in timeline:
                    idx = np.where(timeline == j)[0]
                    prob = KM_estimate[idx]
                surv[i] = prob
            survival.append(surv)

        self.survival_estimate = np.array(survival)

    def predict_batch(self, batch):
        X = batch[:,2:]
        X = torch.from_numpy(X)
        time_output = F.softmax(self.model(X), 1)
        cdf_pred = torch.cumsum(time_output, 1).detach().numpy()
        return cdf_pred
