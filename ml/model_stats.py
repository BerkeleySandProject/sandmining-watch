from torch import nn

def count_number_of_weights(model: nn.Module):
    n_weights_all = 0
    n_weights_trainable = 0

    for param in model.parameters():
        n_weights_all += param.numel()
        if param.requires_grad:
            n_weights_trainable += param.numel()

    return n_weights_all, n_weights_trainable
