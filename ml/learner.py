from typing import (TYPE_CHECKING, Any, Dict, Iterator, List,
                    Optional, Tuple, Union)
from typing_extensions import Literal
from enum import Enum
from abc import ABC
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

from rastervision.core.data import SemanticSegmentationLabels, SemanticSegmentationSmoothLabels
from rastervision.pipeline.file_system import make_dir
from rastervision.pytorch_learner.utils import compute_conf_mat, compute_conf_mat_metrics
from rastervision.pytorch_learner.dataset.visualizer import SemanticSegmentationVisualizer
from rastervision.pytorch_learner.dataset import SemanticSegmentationSlidingWindowGeoDataset

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import Dataset, Sampler

from project_config import CLASS_CONFIG, WANDB_PROJECT_NAME
from experiment_configs.schemas import SupervisedTrainingConfig, BackpropLossChoice
from ml.losses import DiceLoss
from ml.model_stats import count_number_of_weights
from utils.wandb_utils import create_semantic_segmentation_image, create_predicted_probabilities_image
from utils.metrics import compute_metrics
from utils.visualizing import Visualizer

warnings.filterwarnings('ignore')

log = logging.getLogger('rastervision')

MetricDict = Dict[str, float]

"""
Copied and adapted from:
https://github.com/azavea/raster-vision/blob/0.20/rastervision_pytorch_learner/rastervision/pytorch_learner/learner.py
https://github.com/azavea/raster-vision/blob/0.20/rastervision_pytorch_learner/rastervision/pytorch_learner/semantic_segmentation_learner.py
"""

class BinarySegmentationLearner(ABC):
    """Abstract training and prediction routines for a model.

    Hardcoded: Two loss functions, binary cross entropy (BCE) and DICE loss.
    """

    def __init__(self,
                 config: SupervisedTrainingConfig,
                 model: nn.Module,
                 train_ds: 'Dataset',
                 valid_ds: 'Dataset',
                 test_ds: Optional['Dataset'] = None,
                 output_dir: Optional[str] = None,
                 optimizer: Optional['Optimizer'] = None,
                 epoch_scheduler: Optional['_LRScheduler'] = None,
                 step_scheduler: Optional['_LRScheduler'] = None,
                 ):
        self.config = config
        self.device = torch.device('cuda'
                                   if torch.cuda.is_available() else 'cpu')

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        self.model = model
        self.model.to(device=self.device)
        
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(config.mine_class_loss_weight))
        self.dice_loss = DiceLoss()

        self.opt = optimizer
        self.epoch_scheduler = epoch_scheduler
        self.step_scheduler = step_scheduler

        self.num_workers = 0 # 0 means no multiprocessing
        self.train_dl, self.valid_dl, self.test_dl = self.build_dataloaders()

        self.output_dir = output_dir
        if self.output_dir:
            make_dir(self.output_dir)
            self.last_model_weights_path = join(self.output_dir, 'last-model.pth')
        else:
            self.last_model_weights_path = None

        self.class_names = CLASS_CONFIG.names
        self.metric_names = self.build_metric_names()
        self.load_weights_if_available()

        self.visualizer = SemanticSegmentationVisualizer(self.class_names)


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
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler)

        val_dl = DataLoader(
            self.valid_ds,
            batch_size=batch_sz,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True)

        test_dl = None
        if self.test_ds is not None and len(self.test_ds) > 0:
            test_dl = DataLoader(
                self.test_ds,
                batch_size=batch_sz,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True)

        return train_dl, val_dl, test_dl

    def log_data_stats(self):
        """Log stats about each DataSet."""
        if self.train_ds is not None:
            log.info(f'train_ds: {len(self.train_ds)} items')
        if self.valid_ds is not None:
            log.info(f'valid_ds: {len(self.valid_ds)} items')
        if self.test_ds is not None:
            log.info(f'test_ds: {len(self.test_ds)} items')

    def build_step_scheduler(self, start_epoch: int = 0) -> '_LRScheduler':
        """Returns an LR scheduler that changes the LR each step."""
        return None

    def build_metric_names(self) -> List[str]:
        """Returns names of metrics used to validate model at each epoch."""
        metric_names = [
            'epoch', 'train_time', 'valid_time',
            'avg_f1', 'avg_precision', 'avg_recall',
            # In the following, we hardcoded our metric names for our loss functions
            'train_bce_loss', 'val_bce_loss',
            'train_dice_loss', 'val_dice_loss',
        ]

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
        return {"train_bce_loss": self.bce_loss(out, y),
                "train_dice_loss": self.dice_loss(out, y)}

    def validate_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_bce_loss = self.bce_loss(out, y)
        val_dice_loss = self.dice_loss(out, y)
        out = torch.sigmoid(out)

        num_labels = 2 # binary labels
        y = y.view(-1)
        out = self.prob_to_pred(out).view(-1)
        conf_mat = compute_conf_mat(out, y, num_labels)

        # In the following, we hardcoded our metric names for our loss functions
        return {"val_bce_loss": val_bce_loss,
                "val_dice_loss": val_dice_loss,
                 'conf_mat': conf_mat}

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

    def validate_end(self, outputs, num_samples):
        conf_mat = sum([o['conf_mat'] for o in outputs])
        val_bce_loss = torch.stack([o["val_bce_loss"]
                                for o in outputs]).sum() / num_samples
        val_dice_loss = torch.stack([o["val_dice_loss"]
                                for o in outputs]).sum() / num_samples
        conf_mat_metrics = compute_conf_mat_metrics(conf_mat,
                                                    self.class_names)

        metrics = {"val_bce_loss": val_bce_loss.item(),
                   "val_dice_loss": val_dice_loss.item()}
        metrics.update(conf_mat_metrics)

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
        return x.squeeze()

    def prob_to_pred(self, x, threshold=0.5):
        return (x > threshold).int()

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

    def predict(self,
                x: torch.Tensor,
                raw_out: bool = False,
                out_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if out_shape:
            raise ValueError("no support for out_shape")

        x = self.to_batch(x).float()
        x = self.to_device(x, self.device)

        with torch.inference_mode():
            out = self.model(x)
            out = self.post_forward(out)
            out = torch.sigmoid(out)

        if not raw_out:
            out = self.prob_to_pred(out)
        out = self.to_device(out, 'cpu')

        return out
    
    def output_to_numpy(self, out: Tensor) -> np.ndarray:
        """Convert output of model to numpy format.

        Args:
            out: the output of the model in PyTorch format

        Returns: the output of the model in numpy format
        """
        return out.numpy()

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
            collate_fn=self.get_collate_fn(),
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

    def predict_dataloader(
            self,
            dl: DataLoader,
            batched_output: bool = True,
            return_format: Literal['xyz', 'yz', 'z'] = 'z',
            raw_out: bool = True,
            predict_kw: dict = {}
    ) -> Union[Iterator[Any], Iterator[Tuple[Any, ...]]]:
        """Returns an iterator over predictions on the given dataloader.

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
                output_path=join(self.output_dir_local,
                                 'dataloaders/train.png'),
                batch_limit=batch_limit,
                show=show)
        if self.valid_dl:
            log.info('Plotting sample validation batch.')
            self.plot_dataloader(
                self.valid_dl,
                output_path=join(self.output_dir_local,
                                 'dataloaders/valid.png'),
                batch_limit=batch_limit,
                show=show)
        if self.test_dl:
            log.info('Plotting sample test batch.')
            self.plot_dataloader(
                self.test_dl,
                output_path=join(self.output_dir_local,
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
                    # y must be a float to work with PyTorch's BCEWithLogitsLoss
                y = self.to_device(y.float(), self.device)
                batch = (x, y)
                optimizer.zero_grad()
                output = self.train_step(batch, batch_ind)

                if self.config.loss_fn == BackpropLossChoice.DICE:
                    loss_to_backpropagate = output["train_dice_loss"]
                elif self.config.loss_fn == BackpropLossChoice.BCE:
                    loss_to_backpropagate = output["train_bce_loss"]
                else:
                    raise ValueError(f"Unexpected value for loss_fn")
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
                    x = self.to_device(x, self.device)
                    # y must be a float to work with PyTorch's BCEWithLogitsLoss
                    y = self.to_device(y.float(), self.device)
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

    def on_train_start(self):
        """Hook that is called at start of train routine."""
        pass

    def on_epoch_end(self, curr_epoch, metrics):
        """Hook that is called at end of epoch."""
        if self.last_model_weights_path:
            torch.save(self.model.state_dict(), self.last_model_weights_path)

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

    def initialize_wandb_run(self):
        wandb.init(
            project=WANDB_PROJECT_NAME,
            config=self.get_config_dict_for_wandb_log(),
        )
        wandb.define_metric("val_bce_loss", summary="min")
        wandb.define_metric("val_dice_loss", summary="min")
        wandb.define_metric("train_bce_loss", summary="min")
        wandb.define_metric("train_dice_loss", summary="min")
        wandb.define_metric("sandmine_f1", summary="max")
        wandb.define_metric("sandmine_precision", summary="max")
        wandb.define_metric("sandmine_recall", summary="max")
        wandb.watch(self.model, log_freq=100, log_graph=True)

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
    
    def predict_site(
        self,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz = None
    ) -> SemanticSegmentationSmoothLabels:
        predictions = self.predict_dataset(
            ds,
            numpy_out=True,
            progress_bar=False,
        )
        return SemanticSegmentationLabels.from_predictions(
            ds.windows,
            predictions,
            smooth=True,
            extent=ds.scene.extent,
            num_classes=1,
            crop_sz=crop_sz,
        )
    
    def predict_mine_probability_for_site(
        self,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz = None
    ):
        predictions = self.predict_site(ds, crop_sz)
        scores = predictions.get_score_arr(predictions.extent)
        return scores[0]
    
    def evaluate_and_log_to_wandb(self, datasets):
        """
        Runs inference on datasets.
        Logs to W&B:
        - Metrics per observation (unless ground truth is all negative)
        - Metric over all observations
        - RGB image with ground truth and predicted masks per observation (unless ground truth is all negative)
        - Predicted probabilites per observation
        """
        assert wandb.run is not None
        visualizer = Visualizer(self.config.s2_channels)

        segmentation_result_images = []
        predicted_probabilities_images = []
        segmentation_metrics = {}
        predictions_raveled = []
        ground_truths_raveled = []

        for ds in datasets:
            print(ds.scene.id)
            rgb_img = visualizer.rgb_from_bandstack(
                ds.scene.raster_source.get_image_array()
            )
            predicted_probabilities = self.predict_mine_probability_for_site(ds, crop_sz=0)
            predicted_mask = predicted_probabilities > 0.5
            ground_truth_mask = ds.scene.label_source.get_label_arr()

            predictions_raveled.append(predicted_mask.ravel())
            ground_truths_raveled.append(ground_truth_mask.ravel())
            
            predicted_probabilities_images.append(
                create_predicted_probabilities_image(predicted_probabilities, ds.scene.id)
            )
            segmantation_result_image = create_semantic_segmentation_image(
                rgb_img, predicted_mask, ground_truth_mask, ds.scene.id
            )

            ground_truth_is_all_negative = np.all(ground_truth_mask == 0)
            if ground_truth_is_all_negative:
                # If the the observation does not have a labelled mine, it makes no sense to compute our metrics,
                # because we will have no true positives.
                continue

            # Metrics per observation
            precision, recall, f1_score = compute_metrics(ground_truth_mask, predicted_mask)
            segmentation_metrics.update({
                f"eval/{ds.scene.id}/precision": precision,
                f"eval/{ds.scene.id}/recall": recall,
                f"eval/{ds.scene.id}/f1_score": f1_score,
            })
            segmentation_result_images.append(segmantation_result_image)

        # Metrics over all observations
        all_predictions_concatenated  = np.concatenate(predictions_raveled)
        all_ground_truths_concatenated  = np.concatenate(ground_truths_raveled)
        total_precision, total_recall, total_f1_score = compute_metrics(all_ground_truths_concatenated, all_predictions_concatenated)
        wand_log_dict = {
            **segmentation_metrics,
            'eval/total/precision': total_precision,
            'eval/total/recall': total_recall,
            'eval/total/f1_score': total_f1_score,
            'Segmenation masks': segmentation_result_images,
            'Predicted probabilites': predicted_probabilities_images,
        }
        print("Logging evaluation data to W&B")
        wandb.log(wand_log_dict)

    
def get_schedulers_last_lr(scheduler: '_LRScheduler'):
    last_lr = scheduler.get_last_lr()
    if isinstance(last_lr, list) and len(last_lr) == 1:
        return last_lr[0]
    else:
        raise ValueError("Unexpected scheduler.get_last_lr()")