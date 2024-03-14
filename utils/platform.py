import torch


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        if torch.backends.cuda.is_built():
            device = "cuda"
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            device = "cpu"  #"mps"

    return device


def get_accelerator(device):

    accelerator = None

    if device == "cuda":
        accelerator = "gpu"
    elif device == "mps":
        accelerator = "cpu" # "mps"
    else:
        accelerator = "auto"

    return accelerator
