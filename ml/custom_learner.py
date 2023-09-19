import torch
from os.path import join
from rastervision.pytorch_learner import SemanticSegmentationLearner
import wandb
from enum import Enum
import albumentations as A

from ml.model_stats import count_number_of_weights
from project_config import WANDB_PROJECT_NAME

class CustomSemanticSegmentationLearner(SemanticSegmentationLearner):
    """
    Rastervisions SemanticSegmentationLearner class provides a lot the functionalities we need.
    In some cases, we want to customize SemanticSegmentationLearner to our needs, we do this here.
    """
    def __init__(self, experiment_config, **kwargs):
        super().__init__(**kwargs)
        self.experiment_config = experiment_config

    def on_epoch_end(self, curr_epoch, metrics):
        # This funtion extends the regular on_epoch_end() behaviour.
        super().on_epoch_end(curr_epoch, metrics)

        # Log metrics to Weights&Biases
        if wandb.run is not None:
            metrics_to_log = metrics_to_log_wand(metrics)
            wandb.log(metrics_to_log)

        # Default RV saves the model weights to last-model.pth.
        # In the next epoch, RV will overwrite this file.
        # But we want to keep the weights after every epoch
        checkpoint_path = join(self.output_dir_local, f"after-epoch-{curr_epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

    def initialize_wandb_run(self):
        wandb.init(
            project=WANDB_PROJECT_NAME,
            config=self.get_config_dict_for_wandb_log(),
        )
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("sandmine_f1", summary="max")
        #wandb.watch(self.model)

    def get_config_dict_for_wandb_log(self):
        config_to_log = {}
        for key, val in vars(self.experiment_config).items():
            if isinstance(val, Enum):
                config_to_log[key] = val.value
            elif isinstance(val, A.Compose):
                continue
            elif val is None:
                # W&B displays config weirdly when value is None. Therefore we store a string.
                config_to_log[key] = "None"
            else:
                config_to_log[key] = val

        n_weights_total, n_weights_trainable = count_number_of_weights(self.model)
        config_to_log.update(
            {
                'Size training dataset': len(self.train_ds),
                'Size validation dataset': len(self.valid_ds),
                'Number weights total': n_weights_total,
                'Number weights trainable': n_weights_trainable,
            }
        )
        return config_to_log

def metrics_to_log_wand(metrics):
    metrics_to_log = {}
    for key, val in metrics.items():
        if key.startswith('sandmine') or key.endswith('loss'):
            metrics_to_log[key] = val
        elif key.endswith('time'):
            metrics_to_log[f"{key}_per_epoch"] = val
        else:
            continue
    return metrics_to_log
