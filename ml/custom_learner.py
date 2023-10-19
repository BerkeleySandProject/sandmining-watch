from os.path import join
from enum import Enum
from typing import Optional, Dict, Tuple
import numpy as np
import time
import datetime
from tqdm.auto import tqdm
import wandb
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from rastervision.core.data import SemanticSegmentationLabels, SemanticSegmentationSmoothLabels
from rastervision.pytorch_learner import (
    SemanticSegmentationGeoDataConfig, SolverConfig, SemanticSegmentationLearnerConfig,
    SemanticSegmentationSlidingWindowGeoDataset
)
from rastervision.pytorch_learner import SemanticSegmentationLearner
from rastervision.pytorch_learner.utils import compute_conf_mat, compute_conf_mat_metrics

from experiment_configs.schemas import SupervisedTrainingConfig
from ml.model_stats import count_number_of_weights
from utils.visualizing import Visualizer
from utils.wandb_utils import create_semantic_segmentation_image, create_predicted_probabilities_image
from utils.metrics import compute_metrics
from project_config import CLASS_CONFIG, CLASS_NAME, WANDB_PROJECT_NAME

MetricDict = Dict[str, float]

class CustomSemanticSegmentationLearner(SemanticSegmentationLearner):
    """
    Rastervisions SemanticSegmentationLearner class provides a lot the functionalities we need.
    In some cases, we want to customize SemanticSegmentationLearner to our needs, we do this here.
    """
    def __init__(self, 
                 experiment_config, 
                 **kwargs):
        self.experiment_config = experiment_config
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(experiment_config.mine_class_loss_weight))
        self.dice_loss = DiceLoss()
        super().__init__(**kwargs)

    def build_metric_names(self):
        metrics = super().build_metric_names()
        metrics.remove("train_loss")
        metrics.remove("val_loss")
        metrics.append("train_bce_loss")
        metrics.append("train_dice_loss")
        metrics.append("val_bce_loss")
        metrics.append("val_dice_loss")
        return metrics

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

                if self.experiment_config.loss_fn.value is "DICE":
                    loss_to_backpropagate = output["train_dice_loss"]
                elif self.experiment_config.loss_fn.value is "BCE":
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
                else:
                    print("WARNING not step scheduler")
                num_samples += x.shape[0]

        metrics = self.train_end(outputs, num_samples)
        end = time.time()
        metrics['train_time'] = datetime.timedelta(seconds=end - start)
        return metrics
    
    def train_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        return {"train_bce_loss": self.bce_loss(out, y),
                "train_dice_loss": self.dice_loss(out, y)}
    
    def validate_step(self, batch, batch_ind):
        x, y = batch
        out = self.post_forward(self.model(x))
        val_bce_loss = self.bce_loss(out, y)
        val_dice_loss = self.dice_loss(out, y)
        out = torch.sigmoid(out)

        num_labels = len(self.cfg.data.class_names)
        y = y.view(-1)
        out = self.prob_to_pred(out).view(-1)
        conf_mat = compute_conf_mat(out, y, num_labels)

        return {"val_bce_loss": val_bce_loss,
                "val_dice_loss": val_dice_loss,
                 'conf_mat': conf_mat}
    
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
    
    def validate_end(self, outputs, num_samples):
        conf_mat = sum([o['conf_mat'] for o in outputs])
        val_bce_loss = torch.stack([o["val_bce_loss"]
                                for o in outputs]).sum() / num_samples
        val_dice_loss = torch.stack([o["val_dice_loss"]
                                for o in outputs]).sum() / num_samples
        conf_mat_metrics = compute_conf_mat_metrics(conf_mat,
                                                    self.cfg.data.class_names)

        metrics = {"val_bce_loss": val_bce_loss.item(),
                   "val_dice_loss": val_dice_loss.item()}
        metrics.update(conf_mat_metrics)

        return metrics


    def post_forward(self, x):
        # Squeeze to remove the n_classes dimension (since it is size 1)
        # From batch_size x n_classes x width x height
        # To batch_size x width x height
        # Do this to work with PyTorch's BCEWithLogitsLoss
        return super().post_forward(x).squeeze()

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

    def prob_to_pred(self, x, threshold=0.5):
        return (x > threshold).int()

    def on_epoch_end(self, curr_epoch, metrics):
        # This funtion extends the regular on_epoch_end() behaviour.
        super().on_epoch_end(curr_epoch, metrics)

        # Log metrics to Weights&Biases
        if wandb.run is not None:
            metrics_to_log = self.metrics_to_log_wand(metrics)
            # print('logging', metrics_to_log, curr_epoch)
            wandb.log(metrics_to_log)#, step=curr_epoch)

        # Default RV saves the model weights to last-model.pth.
        # In the next epoch, RV will overwrite this file.
        # But we want to keep the weights after every epoch
        #checkpoint_path = join(self.output_dir_local, f"after-epoch-{curr_epoch}.pth")
        #torch.save(self.model.state_dict(), checkpoint_path)

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

    def get_config_dict_for_wandb_log(self):
        config_to_log = {}
        for key, val in vars(self.experiment_config).items():
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
            }
        )
        return config_to_log

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
        visualizer = Visualizer(self.experiment_config.s2_channels)

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
        print("Logging evaluations data to W&B")
        wandb.log(wand_log_dict)

    def metrics_to_log_wand(self, metrics):
        metrics_to_log = {
            'lr_at_epoch_end': get_schedulers_last_lr(self.step_scheduler)
        }
        for key, val in metrics.items():
            if key.startswith('sandmine') or key.endswith('loss'):
                metrics_to_log[key] = val
            elif isinstance(val, datetime.timedelta):
                metrics_to_log[f"{key}_seconds"] = val.total_seconds()
            else:
                continue
        return metrics_to_log


def get_schedulers_last_lr(scheduler: _LRScheduler):
    last_lr = scheduler.get_last_lr()
    if isinstance(last_lr, list) and len(last_lr) == 1:
        return last_lr[0]
    else:
        raise ValueError("Unexpected scheduler.get_last_lr()")


# https://stackoverflow.com/questions/67230305/i-want-to-confirm-which-of-these-methods-to-calculate-dice-loss-is-correct
class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        num = target.shape[0]
        inputs = inputs.reshape(num, -1)
        target = target.reshape(num, -1)

        intersection = (inputs * target).sum(1)
        union = inputs.sum(1) + target.sum(1)
        dice = (2. * intersection) / (union + 1e-8)
        dice = dice.sum()/num
        return 1 - dice
    
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss2(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

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
        lr=config.learning_rate,
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
