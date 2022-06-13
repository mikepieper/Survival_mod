import torch

from models import MLP
from trainer.trainer_base import TrainerBase
from utils.loss import cox_loss_basic, cox_loss_ties
from utils.utils import tgt_equal_tgt, tgt_leq_tgt

import functools
import pdb


class CoxTrainer(TrainerBase):
    
    def __init__(self, cfg):
        """
        Cox trainer.

        Parameters
        ----------
        split : int
            Split number.
        """
        super().__init__(cfg)

        self.model = MLP(cfg, input_shape=self.X_train_shape, output_shape=1)
        if self.use_cuda:
            self.model = self.model.cuda() # f"cuda:{self.cfg.GPU_ID}"
    
        if self.loss_type == "cox_loss_ties":
            self.loss_fn = functools.partial(cox_loss_ties, use_cuda=self.use_cuda)
        elif self.loss_type == "cox_loss_basic":
            self.loss_fn = functools.partial(cox_loss_basic, use_cuda=self.use_cuda)
        else:
            raise ValueError("self.loss_type invalid")


    def process_time(self, time):
        """
        Process batch of time data to extract the explanatory 
        variables matrix and, to compute
        the lower triangle and diagonal by block matrix.

        Parameters
        ----------
        data : ndarray
            Data to process.

        Returns
        -------
        tril : ndarray
            Lower triangular matrix.
        tied_matrix : ndarray
            Diagonal by block matrix.
        """
        time = time.numpy()
        tril = torch.from_numpy(tgt_leq_tgt(time))
        tied_matrix = torch.from_numpy(tgt_equal_tgt(time))
        
        if self.use_cuda:
            tril = tril.cuda()
            tied_matrix = tied_matrix.cuda()

        return tril, tied_matrix


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
        pred = self.model(X)
        concat_pred = torch.cat((time, event, pred), 1)
        tril, tied_matrix = self.process_time(time.cpu())
        loss = self.loss_fn(pred, event, tril, tied_matrix)
        return concat_pred, loss
