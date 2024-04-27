from typing import (TYPE_CHECKING, Any, Dict, Iterator, List,
                    Optional, Tuple, Union)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, Sampler

from experiment_configs.schemas import SupervisedTrainingConfig, ClassConfig, ThreeClassConfig, ThreeClassSupervisedTrainingConfig, ThreeClassVariants
from project_config import ANNO_CONFIG

from torch import nn

from .learner_new import BinarySegmentationLearner, MultiSegmentationLearner

def learner_factory(config: Union[SupervisedTrainingConfig, ThreeClassSupervisedTrainingConfig],
                 model: nn.Module,
                 train_ds: 'Dataset',
                 valid_ds: 'Dataset',
                 output_dir: Optional[str] = None,
                 optimizer: Optional['Optimizer'] = None,
                 epoch_scheduler: Optional['_LRScheduler'] = None,
                 step_scheduler: Optional['_LRScheduler'] = None,
                 save_model_checkpoints = False,
                 load_model_weights=False,
        ):
    if ANNO_CONFIG.num_classes == 2 or (isinstance(config, ThreeClassConfig) and config.three_class_training_method == ThreeClassVariants.A):
        return BinarySegmentationLearner(
            config=config,
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            output_dir=output_dir,
            optimizer=optimizer,
            epoch_scheduler=epoch_scheduler,
            step_scheduler=step_scheduler,
            save_model_checkpoints=save_model_checkpoints,
            load_model_weights=load_model_weights,
        )
    elif ANNO_CONFIG.num_classes > 2:
        return MultiSegmentationLearner(
            config=config,
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            output_dir=output_dir,
            optimizer=optimizer,
            epoch_scheduler=epoch_scheduler,
            step_scheduler=step_scheduler,
            save_model_checkpoints=save_model_checkpoints,
            load_model_weights=load_model_weights,
        )
    else:
        raise ValueError("Learner Choice unknown (invalid number of classes).")
    