from logging import config
from pdb import run
from typing import (TYPE_CHECKING, Any, Dict, Iterator, List,
                    Optional, Tuple, Union)
from typing_extensions import Literal
from enum import Enum
from abc import ABC, abstractmethod
from os.path import join, isfile
import warnings
import time
import datetime
import logging
from pprint import pformat

import numpy as np
from tqdm.auto import tqdm

import wandb
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchmetrics.classification import BinaryAveragePrecision

from rastervision.core.data import SemanticSegmentationLabels, SemanticSegmentationSmoothLabels
from ml.semantic_segmentation_labels import SemanticSegmentationSmoothLabelsCustom, SemanticSegmentationLabelsCustom 
from rastervision.pipeline.file_system import make_dir
from rastervision.pytorch_learner.utils import compute_conf_mat, compute_conf_mat_metrics
from rastervision.pytorch_learner.dataset.visualizer import SemanticSegmentationVisualizer
from rastervision.pytorch_learner.dataset import SemanticSegmentationSlidingWindowGeoDataset
from ml.three_class_metrics import three_class_conf_metrics

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import Dataset, Sampler

from project_config import CLASS_CONFIG, WANDB_PROJECT_NAME, N_EDGE_PIXELS_DISCARD
from experiment_configs.schemas import (SupervisedTrainingConfig, SupervisedFinetuningConfig, 
                                        BackpropLossChoice, FinetuningStratagyChoice, ThreeClassConfig, ThreeClassSupervisedTrainingConfig, ThreeClassVariants)
from ml.losses import DiceLoss, WeightedBCE, WeightedDiceLoss
from ml.model_stats import count_number_of_weights
from utils.wandb_utils import create_semantic_segmentation_image, create_predicted_probabilities_image
from ml.eval_utils import compute_metrics

from peft.peft_model import PeftModel


warnings.filterwarnings('ignore')

log = logging.getLogger('rastervision')

MetricDict = Dict[str, float]

"""
Copied and adapted from:
https://github.com/azavea/raster-vision/blob/0.20/rastervision_pytorch_learner/rastervision/pytorch_learner/learner.py
https://github.com/azavea/raster-vision/blob/0.20/rastervision_pytorch_learner/rastervision/pytorch_learner/semantic_segmentation_learner.py
"""

class Learner(ABC):
    """
    Training routines for a model
    """
    
    def __init__(self,
                 config: Union[SupervisedTrainingConfig, ThreeClassSupervisedTrainingConfig],
                 model: nn.Module,
                 train_ds: 'Dataset',
                 valid_ds: 'Dataset',
                 output_dir: Optional[str] = None,
                 optimizer: Optional['Optimizer'] = None,
                 epoch_scheduler: Optional['_LRScheduler'] = None,
                 step_scheduler: Optional['_LRScheduler'] = None,
                 save_model_checkpoints = False,
                 load_model_weights = False,
                 ):
        self.config = config
        self.device = torch.device('cuda'
                                   if torch.cuda.is_available() else 'cpu')

        self.train_ds = train_ds
        self.valid_ds = valid_ds

        self.model = model
        self.model.to(device=self.device)
        
        self.loss_functions = self.build_loss_functions()
        
        self.opt = optimizer
        self.epoch_scheduler = epoch_scheduler
        self.step_scheduler = step_scheduler

        self.num_workers = 0 # 0 means no multiprocessing
        self.train_dl, self.valid_dl = self.build_dataloaders()

        self.output_dir = output_dir
        if self.output_dir:
            make_dir(self.output_dir)
            self.best_model_weights_path = join(self.output_dir, "best-model.pth")
            self.best_average_precision_score = 0
            self.last_model_weights_path = join(self.output_dir, 'last-model.pth')

            if isinstance(self.config, SupervisedFinetuningConfig) and self.config.finetuning_strategy == FinetuningStratagyChoice.LoRA:
                self.lora_path = join(self.output_dir, "lora")
                make_dir(self.lora_path)

        else:
            self.best_model_weights_path = None
            self.best_average_precision_score = None
            self.last_model_weights_path = None
            
        
        if save_model_checkpoints:
            outdir_dir_folder_name = output_dir.split("/")[-1]
            self.model_checkpoints_dir = f"/data/sand_mining/training_checkpoints/{outdir_dir_folder_name}"
            make_dir(self.model_checkpoints_dir)
            print(f"Will save weights after every epoch to {self.model_checkpoints_dir}")
        else:
            self.model_checkpoints_dir = None

        self.class_names = self.get_class_names()
        self.metric_names = self.build_metric_names()

        if load_model_weights:
            self.load_weights_if_available()

        self.visualizer = SemanticSegmentationVisualizer(self.class_names)
        self.log_data_stats()
        
    def wrapup(self):
        """ Called after all epochs are completed
        """
        if isinstance(self.config, SupervisedFinetuningConfig) and self.config.finetuning_strategy == FinetuningStratagyChoice.LoRA:
            self.merge_lora_weights()

        if self.last_model_weights_path:
            self.save_model_weights(self.last_model_weights_path, False)

    def merge_lora_weights(self):
        print("Merging LoRa weights ...")   
        # print_trainable_parameters(learner.model)

        self.model = self.model.merge_and_unload(safe_merge=True)

    def get_collate_fn(self) -> Optional[callable]:
        """Returns a custom collate_fn to use in DataLoader.

        None is returned if default collate_fn should be used.

        See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        """
        return None

    def get_train_sampler(self, train_ds: 'Dataset') -> Optional['Sampler']:
        """Return a sampler to use for the training dataloader or None to not use any."""
        return None

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Set the DataLoaders for train, validation, and test sets."""

        batch_sz = self.config.batch_size
        collate_fn = self.get_collate_fn()

        train_sampler = self.get_train_sampler(self.train_ds)
        train_shuffle = train_sampler is None
        # batchnorm layers expect batch size > 1 during training
        train_drop_last = (len(self.train_ds) % batch_sz) == 1
        train_dl = DataLoader(
            self.train_ds,
            batch_size=batch_sz,
            shuffle=train_shuffle,
            drop_last=train_drop_last,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=lambda x: torch.multiprocessing.set_sharing_strategy("file_system") if self.num_workers > 0 else None,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler)

        val_dl = DataLoader(
            self.valid_ds,
            batch_size=batch_sz,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=lambda x: torch.multiprocessing.set_sharing_strategy("file_system") if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            pin_memory=True)

        return train_dl, val_dl

    def log_data_stats(self):
        """Log stats about each DataSet."""
        if self.train_ds is not None:
            log.info(f'train_ds: {len(self.train_ds)} items')
        if self.valid_ds is not None:
            log.info(f'valid_ds: {len(self.valid_ds)} items')

    def build_step_scheduler(self, start_epoch: int = 0) -> '_LRScheduler':
        """Returns an LR scheduler that changes the LR each step."""
        return None
    
    def build_metric_names(self) -> List[str]:
        """Returns names of metrics used to validate model at each epoch."""
        metric_names = [
            'epoch', 'train_time', 'valid_time',
            'avg_f1', 'avg_precision', 'avg_recall',
            'sandmine_average_precision'
        ]
        
        for loss_fn in self.loss_functions.keys():
            metric_names.extend([
                f"train_{loss_fn}",
                f"val_{loss_fn}",
            ])

        for label in self.class_names:
            metric_names.extend([
                '{}_f1'.format(label), '{}_precision'.format(label),
                '{}_recall'.format(label)
            ])
        return metric_names
    
    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        # In the following, we hardcoded our metric names for our loss functions
        return self.calculate_losses(out, y, "train_")
    
    def calculate_losses(self, out, y, prefix=""):
        return {
            f"{prefix}{loss_fn_name}": loss_fn(out, y) for loss_fn_name, loss_fn in self.loss_functions.items()
        }
    
    def validate_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        out_probabilities = torch.sigmoid(out)

        return {
            **self.calculate_losses(out, y, "val_"),
            "out_probabilities": out_probabilities.view(-1),
            "ground_truth": y.int().view(-1)
        }

    def train_end(self, outputs: List[MetricDict],
                  num_samples: int) -> MetricDict:
        """Aggregate the ouput of train_step at the end of the epoch.

        Args:
            outputs: a list of outputs of train_step
            num_samples: total number of training samples processed in epoch
        """
        metrics = {}
        for k in outputs[0].keys():
            metrics[k] = torch.stack([o[k] for o in outputs
                                      ]).sum().item() / num_samples
        return metrics

    def plot_dataloader(self,
                        dl: DataLoader,
                        output_path: str,
                        batch_limit: Optional[int] = None,
                        show: bool = False):
        """Plot images and ground truth labels for a DataLoader."""
        x, y = next(iter(dl))
        self.visualizer.plot_batch(
            x, y, output_path, batch_limit=batch_limit, show=show)
        
    def plot_dataloaders(self,
                         batch_limit: Optional[int] = None,
                         show: bool = False):
        """Plot images and ground truth labels for all DataLoaders."""
        if self.train_dl:
            log.info('Plotting sample training batch.')
            self.plot_dataloader(
                self.train_dl,
                output_path=join(self.output_dir,
                                 'dataloaders/train.png'),
                batch_limit=batch_limit,
                show=show)
        if self.valid_dl:
            log.info('Plotting sample validation batch.')
            self.plot_dataloader(
                self.valid_dl,
                output_path=join(self.output_dir,
                                 'dataloaders/valid.png'),
                batch_limit=batch_limit,
                show=show)
        if self.test_dl:
            log.info('Plotting sample test batch.')
            self.plot_dataloader(
                self.test_dl,
                output_path=join(self.output_dir,
                                 'dataloaders/test.png'),
                batch_limit=batch_limit,
                show=show)
            
    def load_weights_if_available(self):
        """Load last weights from previous run if available."""
        weights_path = self.last_model_weights_path
        if weights_path and isfile(weights_path):
            log.info(f'Loading weights from {weights_path}')
            self.model.load_state_dict(
                torch.load(self.last_model_weights_path, map_location=self.device)
            )
            
    def to_device(self, x: Any, device: str) -> Any:
        """Load Tensors onto a device.

        Args:
            x: some object with Tensors in it
            device: 'cpu' or 'cuda'

        Returns:
            x but with any Tensors in it on the device
        """
        if isinstance(x, list):
            return [_x.to(device) if _x is not None else _x for _x in x]
        else:
            return x.to(device)
        
    def process_batch(self, x, y):
        return self.to_device(x, self.device), self.to_device(y, self.device)
        
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
                x, y = self.process_batch(x, y)
                batch = (x, y)
                optimizer.zero_grad()
                output = self.train_step(batch, batch_ind)

                if len(self.loss_functions) == 1:
                    loss_to_backpropagate = list(self.loss_functions.keys())[0]
                else:
                    loss_to_backpropagate = self.config.loss_fn.value
                loss_to_backpropagate = output[f"train_{loss_to_backpropagate}"] 
                loss_to_backpropagate.backward()

                optimizer.step()
                # detach tensors in the output, if any, to avoid memory leaks
                for k, v in output.items():
                    output[k] = v.detach() if isinstance(v, Tensor) else v
                outputs.append(output)
                if step_scheduler is not None:
                    step_scheduler.step()
                num_samples += x.shape[0]

        metrics = self.train_end(outputs, num_samples)
        end = time.time()
        metrics['train_time'] = datetime.timedelta(seconds=end - start)
        return metrics

    def validate_epoch(self, dl: DataLoader) -> MetricDict:
        start = time.time()
        self.model.eval()
        num_samples = 0
        outputs = []
        with torch.inference_mode():
            with tqdm(dl, desc='Validating') as bar:
                for batch_ind, (x, y) in enumerate(bar):
                    x, y = self.process_batch(x, y)
                    batch = (x, y)
                    output = self.validate_step(batch, batch_ind)
                    outputs.append(output)
                    num_samples += x.shape[0]
        metrics = self.validate_end(outputs, num_samples)
        end = time.time()
        metrics['valid_time'] = datetime.timedelta(seconds=end - start)
        return metrics

    def train(self, epochs: Optional[int] = None):
        """Training loop that will attempt to resume training if appropriate."""
        start_epoch = 0
        self.on_train_start()
        for epoch in range(start_epoch, epochs):
            log.info(f'epoch: {epoch}')
            train_metrics = self.train_epoch(
                optimizer=self.opt, step_scheduler=self.step_scheduler)
            if self.epoch_scheduler:
                self.epoch_scheduler.step()
            valid_metrics = self.validate_epoch(self.valid_dl)
            metrics = dict(epoch=epoch, **train_metrics, **valid_metrics)
            log.info(f'metrics:\n{pformat(metrics)}')

            self.on_epoch_end(epoch, metrics)

        self.wrapup()
        
    def on_train_start(self):
        """Hook that is called at start of train routine."""
        pass
    
    def save_model_weights(self, path: str, save_lora=True, curr_epoch=None):
        """Save model weights to path."""
        
        if save_lora and isinstance(self.config, SupervisedFinetuningConfig) and self.config.finetuning_strategy == FinetuningStratagyChoice.LoRA:
            print("Saving LoRa model weights to {}".format(self.lora_path))

            if curr_epoch is not None:
                new_dir = str(curr_epoch)
                new_dir = join(self.lora_path, new_dir)
                make_dir(new_dir)
                self.model.save_pretrained(new_dir)
            else:
                self.model.save_pretrained(self.lora_path)
        else:
            print("Saving final model weights to {}".format(path))
            torch.save(self.model.state_dict(), path)
        
    def save_merged_model(self, path: str):
        #Merge LoRA weights into model
        merged_model = self.model.merge_and_unload()
        #Save using PEFT method
        # merged_model.save_pretrained(path.split("best-model.pth")[0])
        #Save using Torch method
        torch.save(self.merged_model.state_dict(), path)

    def on_epoch_end(self, curr_epoch, metrics):
        """Hook that is called at end of epoch."""
        if self.last_model_weights_path:
            self.save_model_weights(self.last_model_weights_path, curr_epoch=curr_epoch)
            
        if metrics["sandmine_average_precision"] > self.best_average_precision_score:
            print(f"New best average precision score: {metrics['sandmine_average_precision']}.\
                   \nAttempting to save checkpoint to {self.best_model_weights_path}")
            self.best_average_precision_score = metrics["sandmine_average_precision"]
            # self.save_model_weights(self.best_model_weights_path)
            self.save_merged_model(self.best_model_weights_path)

        if self.model_checkpoints_dir:
            torch.save(self.model.state_dict(), f"{self.model_checkpoints_dir}/after_epoch_{curr_epoch}.pth")

        # Log metrics to Weights&Biases
        if wandb.run is not None:
            metrics_to_log = self.metrics_to_log_wand(metrics)
            wandb.log(metrics_to_log)

    def get_config_dict_for_wandb_log(self):
        config_to_log = {}
        for key, val in vars(self.config).items():
            if isinstance(val, Enum):
                config_to_log[key] = val.value
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
                'local_out_dir': self.output_dir,
            }
        )
        return config_to_log
    
    def initialize_wandb_run(self, project=WANDB_PROJECT_NAME,run_name=None, group=None, notes="", resume=False):
        wandb.init(
            project=project,
            config=self.get_config_dict_for_wandb_log(),
            name=run_name,
            group=group,
            entity="sandmining-watch",
            notes=notes,
            resume=resume,
        )
        for loss_fn_name in self.loss_functions.keys():
            wandb.define_metric(f"val_{loss_fn_name}", summary="min")
        wandb.define_metric("sandmine_f1", summary="max")
        wandb.define_metric("sandmine_average_precision", summary="max")
        wandb.watch(self.model, log_freq=100, log_graph=True)
        print("W&B ID: ", wandb.run.id)
        
    def metrics_to_log_wand(self, metrics):
        metrics_to_log = {}
        if self.step_scheduler is not None:
            metrics_to_log['lr_at_epoch_end'] = get_schedulers_last_lr(self.step_scheduler)
        for key, val in metrics.items():
            if key.startswith('sandmine') or key.endswith('loss'):
                metrics_to_log[key] = val
            elif isinstance(val, datetime.timedelta):
                metrics_to_log[f"{key}_seconds"] = val.total_seconds()
            else:
                continue
        return metrics_to_log
    
    @abstractmethod
    def out_to_prob(self, out):
        pass
    
    @abstractmethod
    def validate_end(self, outputs, num_samples):
        pass
    
    @abstractmethod
    def post_forward(self, x: Any) -> Any:
        """Post process output of call to model().
        Useful for when predictions are inside a structure returned by model().
        """
        pass
    
    @abstractmethod
    def prob_to_pred(self, x, threshold=0.5):
        pass
    
    @abstractmethod
    def build_loss_functions(self) -> dict:
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        pass
    
    
    
def get_schedulers_last_lr(scheduler: '_LRScheduler'):
    last_lr = scheduler.get_last_lr()
    if isinstance(last_lr, list) and len(last_lr) == 1:
        return last_lr[0]
    else:
        raise ValueError("Unexpected scheduler.get_last_lr()")
    
class BinarySegmentationLearner(Learner):
    
    def build_loss_functions(self) -> dict:
        if isinstance(self.config, ThreeClassConfig) and self.config.three_class_training_method == ThreeClassVariants.A:
            return {
                "bce_loss": WeightedBCE(
                    nonmine_weight=1.0,
                    mine_weight=self.config.mine_class_loss_weight
                ),
                "dice_loss": WeightedDiceLoss()
            }
        else:
            return {
                "bce_loss": nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(self.config.mine_class_loss_weight)),
                "dice_loss": DiceLoss()
            }
            
    def get_class_names(self) -> List[str]:
        if isinstance(self.config, ThreeClassConfig) and self.config.three_class_training_method == ThreeClassVariants.A:
            return [CLASS_CONFIG.names[0], CLASS_CONFIG.names[2]]
        else:
            return CLASS_CONFIG.names
            
    def out_to_prob(self, out):
        return torch.sigmoid(out)
    
    def validate_end(self, outputs, num_samples):
        def concat_values(key):
            return torch.cat([o[key] for o in outputs])
        def stack_values(key):
            return torch.stack([o[key] for o in outputs])
        
        out_probabilities = concat_values("out_probabilities")
        ground_truths = concat_values("ground_truth")
        stacked_losses = {
            loss_fn_name: stack_values(loss_fn_name) 
            for loss_fn_name in outputs[0].keys() 
            if loss_fn_name[4:] in list(self.loss_functions.keys())
        }

        if isinstance(self.config, ThreeClassConfig) and self.config.three_class_training_method == ThreeClassVariants.A:
            mask = ground_truths != 1
            out_probabilities = out_probabilities[mask]     # Filtering out low confidence
            ground_truths = ground_truths[mask]             # ""
            ground_truths[ground_truths == 2] = 1           # Assigning high confidence to value 1 to work with binary outputs
        
        out_classes =  self.prob_to_pred(out_probabilities)
        conf_mat = compute_conf_mat(
            out_classes,
            ground_truths,
            num_labels=2, # we have two classes
        )
        conf_mat_metrics = compute_conf_mat_metrics(conf_mat, self.class_names)

        _,_,_, average_precision, best_threshold, best_f1_score = compute_metrics(ground_truths.cpu().numpy(), out_probabilities.cpu().numpy())

        metrics = {
            **conf_mat_metrics,
            "sandmine_average_precision": average_precision,
            "sandmine_best_threshold": best_threshold,
            "sandmine_best_f1_score": best_f1_score,
            **{
                loss_fn_name: loss_values.cpu().numpy().sum() / num_samples
                for loss_fn_name, loss_values in stacked_losses.items()
            },
        }
        return metrics
    
    def post_forward(self, x: Any) -> Any:
        """Post process output of call to model().
        Useful for when predictions are inside a structure returned by model().
        """
        if isinstance(x, dict):
            x = x['out']
        # Squeeze to remove the n_classes dimension (since it is size 1)
        # From batch_size x n_classes x width x height
        # To batch_size x width x height
        # Do this to work with PyTorch's BCEWithLogitsLoss
        return x.squeeze(dim=1)
    
    def prob_to_pred(self, x, threshold=0.5):
        return (x > threshold).int()
    
    def process_batch(self, x, y):
        x, y = super().process_batch(x, y)
        return x, y.float()

class MultiSegmentationLearner(Learner):
    def build_loss_functions(self) -> dict:
        return {
            "ce_loss": nn.CrossEntropyLoss(weight=self.to_device(torch.tensor([1., self.config.low_confidence_weight, self.config.mine_class_loss_weight]), self.device))
        }
        
    def out_to_prob(self, out):
        return out.softmax(dim=1)
    
    def post_forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            return x['out']
        return x
    
    def prob_to_pred(self, x):
        return x.argmax(1)
    
    def get_class_names(self) -> List[str]:
        return CLASS_CONFIG.names
    
    def validate_end(self, outputs, num_samples):
        # outputs is a list of dictionaries.
        # Each element in the list corresponds to outputs from one validation batch

        def concat_values(key):
            return torch.cat([o[key] for o in outputs])
        def stack_values(key):
            return torch.stack([o[key] for o in outputs])
        
        out_probabilities = concat_values("out_probabilities")
        ground_truths = concat_values("ground_truth")
        stacked_losses = {
            loss_fn_name: stack_values(loss_fn_name) 
            for loss_fn_name in outputs[0].keys() 
            if loss_fn_name[4:] in list(self.loss_functions.keys())
        }

        out_classes =  self.prob_to_pred(out_probabilities.view(num_samples, 3, self.config.tile_size, self.config.tile_size)).reshape(-1)
        conf_mat = compute_conf_mat(
            out_classes,
            ground_truths,
            num_labels=3,
        )
        conf_mat_metrics = compute_conf_mat_metrics(conf_mat, self.class_names)

        # bap = BinaryAveragePrecision(thresholds=None)
        # average_precision1 = bap(out_probabilites, ground_truths)
        high_conf_probs = out_probabilities.reshape(num_samples, 3, self.config.tile_size, self.config.tile_size)[:, 2, :, :]     # Select only high confidence sandmine
        high_conf_probs = high_conf_probs.reshape(-1)
        mask = ground_truths != 1
        high_conf_probs = high_conf_probs[mask]       # Filtering out low confidence
        predicted_classes = out_classes[mask]         # ""
        ground_truths = ground_truths[mask]           # ""
        predicted_classes[predicted_classes == 2] = 1               # Assigning high confidence to value 1 to work with binary outputs
        ground_truths[ground_truths == 2] = 1                       # "" 

        _,_,_, average_precision, best_threshold, best_f1_score = compute_metrics(ground_truths.cpu().numpy(), high_conf_probs.cpu().numpy(), predicted_classes.cpu().numpy())

        metrics = {
            **conf_mat_metrics,
            "sandmine_average_precision": average_precision,
            "sandmine_best_threshold": best_threshold,
            "sandmine_best_f1_score": best_f1_score,
            **{
                loss_fn_name: loss_values.cpu().numpy().sum() / num_samples
                for loss_fn_name, loss_values in stacked_losses.items()
            },
        }
        return metrics
    
    def calculate_losses(self, out, y, prefix=""):
        y = y.long()
        return super().calculate_losses(out, y, prefix)
    

########################
# Predictor            #
########################

class Predictor(ABC):
    """
    Prediction routines for a model.
    """

    def __init__(self,
                 config: SupervisedTrainingConfig,
                 model: nn.Module,
                 path_to_weights: Optional[str] = None,
                 path_to_lora: Optional[str] = None,
                 temperature: float = None
                 ):
        self.config = config
        self.device = torch.device('cuda'
                                   if torch.cuda.is_available() else 'cpu')
        
        self.model = model
        self.model.to(device=self.device)
        
        if path_to_weights and path_to_lora: #then only load decoder weights from path_to_weight, and then use lora
            print("(Predictor) Loading decoder weights from {}".format(path_to_weights))
            self.model.load_decoder_weights(path_to_weights)

            print("(Predictor) Loading LoRa weights from {}".format(path_to_lora))
            self.model = PeftModel.from_pretrained(self.model, path_to_lora)
        elif path_to_weights: #load encoder and decoder from path_to_weights
            print("(Predictor) Loading ALL weights from {}".format(path_to_weights))
            self.model.load_state_dict(
                torch.load(path_to_weights, map_location=self.device)
            )

        self.class_names = self.get_class_names()
        self.num_workers = 0 # 0 means no multiprocessing

        self._set_temperature(temperature)

    def _set_temperature(self, 
                         temperature: float = None):
        
        self.temperature = temperature
        if self.temperature is None:
            print("Temperature scaling set to None")
        else:
            print("Predictor has a calibration temperature of {}".format(self.temperature))

    def predict_dataset(self,
                        dataset: 'Dataset',
                        return_format: Literal['xyz', 'yz', 'z'] = 'z',
                        raw_out: bool = True,
                        numpy_out: bool = False,
                        predict_kw: dict = {},
                        dataloader_kw: dict = {},
                        progress_bar: bool = True,
                        progress_bar_kw: dict = {}
                        ) -> Union[Iterator[Any], Iterator[Tuple[Any, ...]]]:
        """Returns an iterator over predictions on the given dataset.

        Args:
            dataset (Dataset): The dataset to make predictions on.
            return_format (Literal['xyz', 'yz', 'z'], optional): Format of the
                return elements of the returned iterator. Must be one of:
                'xyz', 'yz', and 'z'. If 'xyz', elements are 3-tuples of x, y,
                and z. If 'yz', elements are 2-tuples of y and z. If 'z',
                elements are (non-tuple) values of z. Where x = input image,
                y = ground truth, and z = prediction. Defaults to 'z'.
            raw_out (bool, optional): If true, return raw predicted scores.
                Defaults to True.
            numpy_out (bool, optional): If True, convert predictions to numpy
                arrays before returning. Defaults to False.
            predict_kw (dict): Dict with keywords passed to Learner.predict().
                Useful if a Learner subclass implements a custom predict()
                method.
            dataloader_kw (dict): Dict with keywords passed to the DataLoader
                constructor.
            progress_bar (bool, optional): If True, display a progress bar.
                Since this function returns an iterator, the progress bar won't
                be visible until the iterator is consumed. Defaults to True.
            progress_bar_kw (dict): Dict with keywords passed to tqdm.

        Raises:
            ValueError: If return_format is not one of the allowed values.

        Returns:
            Union[Iterator[Any], Iterator[Tuple[Any, ...]]]: If return_format
                is 'z', the returned value is an iterator of whatever type the
                predictions are. Otherwise, the returned value is an iterator
                of tuples.
        """

        if return_format not in {'xyz', 'yz', 'z'}:
            raise ValueError('return_format must be one of "xyz", "yz", "z".')

        dl_kw = dict(
            collate_fn=None,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True)
        dl_kw.update(dataloader_kw)
        dl = DataLoader(dataset, **dl_kw)

        preds = self.predict_dataloader(
            dl,
            return_format=return_format,
            raw_out=raw_out,
            batched_output=False,
            predict_kw=predict_kw)

        if numpy_out:
            if return_format == 'z':
                preds = (self.output_to_numpy(p) for p in preds)
            else:
                # only convert z
                preds = ((*p[:-1], self.output_to_numpy(p[-1])) for p in preds)

        if progress_bar:
            pb_kw = dict(desc='Predicting', total=len(dataset))
            pb_kw.update(progress_bar_kw)
            preds = tqdm(preds, **pb_kw)

        return preds

    def predict(self,
                x: torch.Tensor,
                raw_out: bool = False,
                out_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        asd
        """
        if out_shape:
            raise ValueError("no support for out_shape")

        x = self.to_batch(x).float()
        x = self.to_device(x, self.device)

        with torch.inference_mode():
            out = self.model(x)
            out = self.post_forward(out)
            out = self.out_to_prob(out)

        if not raw_out:
            out = self.prob_to_pred(out)
        out = self.to_device(out, 'cpu')

        return out
    
    def predict_dataloader(
            self,
            dl: DataLoader,
            batched_output: bool = True,
            return_format: Literal['xyz', 'yz', 'z'] = 'z',
            raw_out: bool = True,
            predict_kw: dict = {}
    ) -> Union[Iterator[Any], Iterator[Tuple[Any, ...]]]:
        """
        Returns an iterator over predictions on the given dataloader.

        Args:
            dl (DataLoader): The dataloader to make predictions on.
            batched_output (bool, optional): If True, return batches of
                x, y, z as defined by the dataloader. If False, unroll the
                batches into individual items. Defaults to True.
            return_format (Literal['xyz', 'yz', 'z'], optional): Format of the
                return elements of the returned iterator. Must be one of:
                'xyz', 'yz', and 'z'. If 'xyz', elements are 3-tuples of x, y,
                and z. If 'yz', elements are 2-tuples of y and z. If 'z',
                elements are (non-tuple) values of z. Where x = input image,
                y = ground truth, and z = prediction. Defaults to 'z'.
            raw_out (bool, optional): If true, return raw predicted scores.
                Defaults to True.
            predict_kw (dict): Dict with keywords passed to Learner.predict().
                Useful if a Learner subclass implements a custom predict()
                method.

        Raises:
            ValueError: If return_format is not one of the allowed values.

        Returns:
            Union[Iterator[Any], Iterator[Tuple[Any, ...]]]: If return_format
                is 'z', the returned value is an iterator of whatever type the
                predictions are. Otherwise, the returned value is an iterator
                of tuples.
        """
        if return_format not in {'xyz', 'yz', 'z'}:
            raise ValueError('return_format must be one of "xyz", "yz", "z".')

        preds = self._predict_dataloader(
            dl,
            raw_out=raw_out,
            batched_output=batched_output,
            predict_kw=predict_kw)

        if return_format == 'yz':
            preds = ((y, z) for _, y, z in preds)
        elif return_format == 'z':
            preds = (z for _, _, z in preds)

        return preds

    def _predict_dataloader(
            self,
            dl: DataLoader,
            raw_out: bool = True,
            batched_output: bool = True,
            predict_kw: dict = {}) -> Iterator[Tuple[Tensor, Any, Any]]:
        """Returns an iterator over predictions on the given dataloader.

        Args:
            dl (DataLoader): The dataloader to make predictions on.
            batched_output (bool, optional): If True, return batches of
                x, y, z as defined by the dataloader. If False, unroll the
                batches into individual items. Defaults to True.
            raw_out (bool, optional): If true, return raw predicted scores.
                Defaults to True.
            predict_kw (dict): Dict with keywords passed to Learner.predict().
                Useful if a Learner subclass implements a custom predict()
                method.

        Raises:
            ValueError: If return_format is not one of the allowed values.

        Yields:
            Iterator[Tuple[Tensor, Any, Any]]: 3-tuples of x, y, and z, which
                might or might not be batched depending on the batched_output
                argument.
        """
        self.model.eval()

        for x, y in dl:
            x = self.to_device(x, self.device)
            z = self.predict(x, raw_out=raw_out, **predict_kw)
            x = self.to_device(x, 'cpu')
            y = self.to_device(y, 'cpu') if y is not None else y
            z = self.to_device(z, 'cpu')
            if batched_output:
                yield x, y, z
            else:
                for _x, _y, _z in zip(x, y, z):
                    yield _x, _y, _z


    def output_to_numpy(self, out: Tensor) -> np.ndarray:
        """Convert output of model to numpy format.

        Args:
            out: the output of the model in PyTorch format

        Returns: the output of the model in numpy format
        """
        return out.numpy()

    def to_batch(self, x: Tensor) -> Tensor:
        """Ensure that image array has batch dimension.

        Args:
            x: assumed to be either image or batch of images

        Returns:
            x with extra batch dimension of length 1 if needed
        """
        if x.ndim == 3:
            x = x[None, ...]
        return x
    
    def to_device(self, x: Any, device: str) -> Any:
        """Load Tensors onto a device.

        Args:
            x: some object with Tensors in it
            device: 'cpu' or 'cuda'

        Returns:
            x but with any Tensors in it on the device
        """
        if isinstance(x, list):
            return [_x.to(device) if _x is not None else _x for _x in x]
        else:
            return x.to(device)

    def predict_site(
        self,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz: int,
        num_classes=1,
    ) -> SemanticSegmentationSmoothLabels:
        
        predictions_on_dl = self.predict_dataset( #Each 160x160 (or other sized) images are predicted
            ds,
            numpy_out=True,
            progress_bar=False,
        )

        predictions_site =  SemanticSegmentationLabelsCustom.from_predictions(  #this is where different predicition outputs in a scene are merged
            ds.windows,
            predictions_on_dl,
            # smooth=True,
            extent=ds.scene.extent,
            num_classes=num_classes,
            crop_sz=crop_sz,
            apply_kernel=self.config.apply_smoothing,
            tile_size = self.config.tile_size,
            sigma = self.config.smoothing_sigma,
        )

        return predictions_site

    def predict_mine_probability_for_site(
        self,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz = N_EDGE_PIXELS_DISCARD,
    ):
        predictions = self.predict_site(ds, crop_sz)
        scores = predictions.get_score_arr(predictions.extent)
        return scores[0]
    
    @abstractmethod
    def post_forward(self, x: Any) -> Any:
        """Post process output of call to model().
        Useful for when predictions are inside a structure returned by model().
        """
        pass
    
    @abstractmethod
    def out_to_prob(self, out):
        pass
    
    @abstractmethod
    def prob_to_pred(self, out):
        pass

class BinarySegmentationPredictor(Predictor):
    def post_forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            x = x['out']
        # Squeeze to remove the n_classes dimension (since it is size 1)
        # From batch_size x n_classes x width x height
        # To batch_size x width x height
        x =  x.squeeze(1)
        if self.temperature is not None:
            x /= self.temperature

        return x
    
    def out_to_prob(self, out):
        return torch.sigmoid(out)
    
    def prob_to_pred(self, out, threshold=.5):
        return (out > threshold).int()
    
    def get_class_names(self):
        if isinstance(self.config, ThreeClassConfig) and self.config.three_class_training_method == ThreeClassVariants.A:
            return [CLASS_CONFIG.names[0], CLASS_CONFIG.names[2]]
        else:
            return CLASS_CONFIG.names
    
class MultiSegmentationPredictor(Predictor):
    def post_forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            x = x['out']
        if self.temperature is not None:
            x /= self.temperature
        return x
    
    def out_to_prob(self, out):
        return torch.softmax(out, dim=1)
    
    def prob_to_pred(self, out):
        return out.argmax(1)
    
    def get_class_names(self):
        return CLASS_CONFIG.names
    
    def predict_site(
        self,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz: int,
    ) -> SemanticSegmentationSmoothLabels:
        return super().predict_site(
            ds, crop_sz, num_classes=3
        )
    