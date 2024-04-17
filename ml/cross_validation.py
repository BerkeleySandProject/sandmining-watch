from utils.rastervision_pipeline import scene_to_training_ds, scene_to_inference_ds
from torch.utils.data import ConcatDataset
from sklearn.model_selection import GroupKFold, LeavePGroupsOut
import numpy as np

from models.model_factory import model_factory, print_trainable_parameters
from ml.optimizer_factory import optimizer_factory
from ml.learner_factory import learner_factory

import wandb
import gc

import torch
import random

from os.path import expanduser




class CrossValidator:
    def __init__(self, scenes, cluster_ids, split_groups=None, num_splits=None, size_validation_group=None) -> None:
        """
        split_groups is for manually assigning splits. Should be a list (number of splits in length) containing a list of 
        training and validation cluster ids. i.e. [([1, 2], [3]), ([4, 5], [6])].
        
        num_splits is the number of splits
        
        size_validation_group is used for leave p groups validation set and the rest in the training set
        """
        assert (split_groups is not None) ^ (num_splits is not None) ^ (size_validation_group is not None), "Only one of splits, num_splits, size_validation_group should not be None"
        
        self.scenes = scenes
        self.cluster_ids = np.array(cluster_ids)
        self.splits = None
        self.split_groups = split_groups
        self.num_splits = num_splits
        
        if self.split_groups is not None:
            self.splits = []
            for split in self.split_groups:
                train_split = []
                for cid in split[0]:
                    assert cid in self.cluster_ids, f"Training Cluster {cid} not in the available clusters"
                    train_split += [i for i in range(len(self.cluster_ids)) if self.cluster_ids[i] == cid]
                val_split = []
                for cid in split[1]:
                    assert cid in self.cluster_ids, f"Validation Cluster {cid} not in the available clusters"
                    val_split += [i for i in range(len(self.cluster_ids)) if self.cluster_ids[i] == cid]
                self.splits.append((np.array(train_split), np.array(val_split)))
        elif self.num_splits is not None:
            gkf = GroupKFold(self.num_splits)
            scenes = list(gkf.split(np.arange(len(self.cluster_ids)), groups=self.cluster_ids))
            self.splits = scenes
            self.split_groups = []
            for split in self.splits:
                train_cids = np.unique(self.cluster_ids[split[0]])
                val_cids = np.unique(self.cluster_ids[split[1]])
                self.split_groups.append((train_cids, val_cids))
        else:   # size_validation_group is not None
            lpgo = LeavePGroupsOut(n_groups=size_validation_group)
            scenes = list(lpgo.split(np.arange(len(self.cluster_ids)), groups=self.cluster_ids))
            self.splits = scenes
            self.split_groups = []
            for split in self.splits:
                train_cids = np.unique(self.cluster_ids[split[0]])
                val_cids = np.unique(self.cluster_ids[split[1]])
                self.split_groups.append((train_cids, val_cids))
        self.num_splits = len(self.splits)
        
    def _train(self, model_config, lora_config, train_split, val_split, num_epochs, wandb_group_name, run_name, model_weights_output_folder):
        #For reproducibility
        torch.manual_seed(13)
        random.seed(13)
        
        train_ds = [scene_to_training_ds(model_config, self.scenes[sid]) for sid in train_split]
        valid_ds = [scene_to_inference_ds(model_config, self.scenes[sid], full_image=False, stride=int(config.tile_size/2)) for sid in val_split]
        train_ds_merged = ConcatDataset(train_ds)
        valid_ds_merged = ConcatDataset(valid_ds)
        
        # Here to avoid cuda out of memory errors
        torch.cuda.empty_cache()
        
        _, _, n_channels = train_ds[0].scene.raster_source.shape
        model = model_factory(
            model_config,
            n_channels=n_channels,
            config_lora=lora_config
        )
        
        output_dir = expanduser(model_weights_output_folder + run_name) if model_weights_output_folder is not None else None
        
        optimizer = optimizer_factory(model_config, model)
        learner = learner_factory(
            config=model_config,
            model=model,
            optimizer=optimizer,
            train_ds=train_ds_merged,  # for development and debugging, use training_datasets[0] or similar to speed up
            valid_ds=valid_ds_merged,  # for development and debugging, use training_datasets[1] or similar to speed up
            output_dir=output_dir,
        )
        
        print_trainable_parameters(learner.model)
        learner.initialize_wandb_run(run_name=run_name, group=wandb_group_name)
        learner.train(epochs=num_epochs)
        wandb.finish()
        
        # Here to avoid cuda out of memory errors
        del learner
        del model
        del train_ds
        del valid_ds
        gc.collect()
        torch.cuda.empty_cache()
        
    def cross_val(self, model_config, num_epochs, lora_config=None, wandb_group_name=None, model_weights_output_folder=None):
        for i, (train_split, valid_split) in enumerate(self.splits):
            self._train(model_config, 
                        lora_config, 
                        train_split, 
                        valid_split, 
                        num_epochs,
                        wandb_group_name,
                        "val_cids_" + "_".join([str(j) for j in self.split_groups[i][1]]),
                        model_weights_output_folder)
       