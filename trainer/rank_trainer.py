import torch

from models import MLP
from trainer.trainer_base import TrainerBase
from utils.config import cfg
from utils.loss import cox_loss_basic, cox_loss_ties, rank_loss


class RankTrainer(TrainerBase):
    def __init__(self, cfg, split):
        """
        Likelihood trainer.
        """
        super().__init__(cfg, split)

        self.model = MLP(cfg, input_shape=self.X_train_shape, output_shape=1)
        if self.use_cuda:
            self.model.cuda()

        self.rank_function = cfg.TRAIN.F_RANK

    def get_pred_loss(self, batch):
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
        time, event, X, _, _ = self.process_batch(batch)

        # Creating tensors
        time = torch.from_numpy(time)
        event = torch.from_numpy(event)
        if cfg.DATA.DEATH_AT_CENSOR_TIME:
            event = torch.ones(event.size())
        X = torch.from_numpy(X)
        
        if self.use_cuda:
            time = time.cuda()
            event = event.cuda()
            X = X.cuda()

        pred = self.model(X)

        concat_pred = torch.cat((time.unsqueeze(-1), event.unsqueeze(-1)), 1)
        concat_pred = torch.cat((concat_pred, pred), 1)

        loss = rank_loss(pred, time, event, f=self.rank_function)

        return concat_pred, loss
