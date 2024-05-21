from torch import nn
import torch.optim as optim

# from models.satmae.util.lr_decay import param_groups_lrd_2
import ipdb
from experiment_configs.schemas import SupervisedTrainingConfig, OptimizerChoice


def optimizer_factory(
    config: SupervisedTrainingConfig, model: nn.Module
) -> optim.Optimizer:
    if config.optimizer == OptimizerChoice.AdamW:
        ipdb.set_trace()
        # DEBUG: Check that optimizer only optimzes certain specified layers
        # return optim.AdamW(model.parameters(), lr=config.learning_rate)
        return optim.AdamW(
            filter(lambda p: p.requires_grad is True, model.parameters()),
            lr=config.learning_rate,
        )
    elif config.optimizer == OptimizerChoice.SDG:
        return optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError("Optimizer Choice unknown.")


# def get_optimizer(config, model):

#     param_groups_encoder = param_groups_lrd_2(
#         model.encoder,
#         max_lr=config.learning_rate,
#         weight_decay=0.05,
#     )
#     param_group_decoder = {
#         "lr": config.learning_rate,
#         "weight_decay": 0.05,
#         "params": model.decoder.parameters(),
#     }

#     optimizer = optim.SGD(
#         [param_groups_encoder, param_group_decoder],
#         #lr=config.learning_rate,
#         #momentum=0.9,
#         #weight_decay=1e-4
#     )
#     return optimizer
