from os.path import join
from enum import Enum
from typing import Optional, Dict
import time
import datetime
from tqdm.auto import tqdm
import wandb
import albumentations as A
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from rastervision.pytorch_learner import (
    SemanticSegmentationGeoDataConfig, SolverConfig, SemanticSegmentationLearnerConfig
)
from rastervision.pytorch_learner import SemanticSegmentationLearner

from experiment_configs.schemas import SupervisedTrainingConfig
from ml.model_stats import count_number_of_weights
from project_config import CLASS_CONFIG, WANDB_PROJECT_NAME  

MetricDict = Dict[str, float]

class CustomSemanticSegmentationLearner(SemanticSegmentationLearner):
    """
    Rastervisions SemanticSegmentationLearner class provides a lot the functionalities we need.
    In some cases, we want to customize SemanticSegmentationLearner to our needs, we do this here.
    """
    def __init__(self, experiment_config, **kwargs):
        super().__init__(**kwargs)
        self.experiment_config = experiment_config

    def train_epoch(
            self,
            optimizer: 'Optimizer',
            step_scheduler: Optional['_LRScheduler'] = None) -> MetricDict:
        """Train for a single epoch. Overwriting RV's Learner function."""
        start = time.time()
        self.model.train()
        num_samples = 0
        outputs = []
        with tqdm(self.train_dl, desc='Training') as bar:
            for batch_ind, (x, y) in enumerate(bar):
                x = self.to_device(x, self.device)
                y = y.to(torch.int64)
                y = self.to_device(y, self.device)
                batch = (x, y)
                optimizer.zero_grad()
                output = self.train_step(batch, batch_ind)
                output['train_loss'].backward()
                optimizer.step()
                # detach tensors in the output, if any, to avoid memory leaks
                for k, v in output.items():
                    output[k] = v.detach() if isinstance(v, Tensor) else v
                outputs.append(output)
                if step_scheduler is not None:
                    step_scheduler.step()
                else:
                    print("WARNING not step scheduler")
                num_samples += x.shape[0]

                # if wandb.run is not None:
                #     last_lr = step_scheduler.get_last_lr()
                #     if isinstance(last_lr, list) and len(last_lr) == 1:
                #         print("last_lr[0]", last_lr[0])
                #         wandb.log({'lr': last_lr[0]})
                #     else:
                #         print("Unexpected scheduler.get_last_lr()")

        metrics = self.train_end(outputs, num_samples)
        end = time.time()
        train_time = datetime.timedelta(seconds=end - start)
        metrics['train_time'] = str(train_time)
        return metrics


    def on_epoch_end(self, curr_epoch, metrics):
        # This funtion extends the regular on_epoch_end() behaviour.
        super().on_epoch_end(curr_epoch, metrics)

        # Log metrics to Weights&Biases
        if wandb.run is not None:
            metrics_to_log = metrics_to_log_wand(metrics)
            # print('logging', metrics_to_log, curr_epoch)
            wandb.log(metrics_to_log)#, step=curr_epoch)

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
        wandb.watch(self.model, log_freq=100, log_graph=True)

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



def learner_factory(
        config: SupervisedTrainingConfig,
        model: nn.Module,
        optimizer: Optimizer,
        training_ds: Dataset,
        validation_ds: Dataset,
) -> CustomSemanticSegmentationLearner:
    # If a last-model.pth exists in experiment_dir, the learner will load its weights. 
    data_cfg = SemanticSegmentationGeoDataConfig(
        class_names=CLASS_CONFIG.names,
        class_colors=CLASS_CONFIG.colors,
        num_workers=0,
    )
    solver_cfg = SolverConfig(
        batch_sz=config.batch_size,
        class_loss_weights=[1., config.mine_class_loss_weight]
    )
    learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)
    
    learner = CustomSemanticSegmentationLearner(
        optimizer=optimizer,
        experiment_config=config,
        cfg=learner_cfg,
        output_dir=config.output_dir,
        model=model,
        train_ds=training_ds,
        valid_ds=validation_ds,
        training=True,
    )
    assert learner.epoch_scheduler is None 
    return learner
