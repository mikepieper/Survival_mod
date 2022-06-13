from .cox_trainer import CoxTrainer
from .emd_trainer import EMDTrainer
from .rank_trainer import RankTrainer


def get_algo(cfg):
    if cfg.TRAIN.MODEL == "cox":
        return CoxTrainer(cfg)
    elif cfg.TRAIN.MODEL == "emd":
        return EMDTrainer(cfg)
    elif cfg.TRAIN.MODEL == "rank":
        return RankTrainer(cfg)
    else:
        raise NotImplementedError()
