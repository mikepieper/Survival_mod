from .cox_trainer import CoxTrainer
from .emd_trainer import EMDTrainer
from .rank_trainer import RankTrainer


def get_algo(cfg, split):
    if cfg.TRAIN.MODEL == "cox":
        return CoxTrainer(cfg, split)
    elif cfg.TRAIN.MODEL == "emd":
        return EMDTrainer(cfg, split)
    elif cfg.TRAIN.MODEL == "rank":
        return RankTrainer(cfg, split)
    else:
        raise NotImplementedError()
