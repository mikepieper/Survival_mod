import torch

from models import MLP
from trainer.trainer_base import TrainerBase
from utils.loss import rank_loss
import functools
import pdb


class RankTrainer(TrainerBase):
    
    
    def __init__(self, cfg):
        """
        Likelihood trainer.

        Parameters
        ----------
        split : int
            Split number.
        """
        super().__init__(cfg)
        self.model = MLP(cfg, input_shape=self.X_train_shape, output_shape=1)
        if self.use_cuda:
            self.model = self.model.cuda()

        self.loss_fn = functools.partial(rank_loss, f=self.cfg.TRAIN.F_RANK, use_cuda=self.use_cuda)


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

        loss = self.loss_fn(pred, time, event)
        return concat_pred, loss
