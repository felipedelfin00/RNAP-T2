import contextlib
import numpy as np
import random
import torch

from .config import DETERMINISTIC, USE_AMP


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cudnn.benchmark = not DETERMINISTIC


def get_amp_dtype(device):
    if not USE_AMP or device.type != "cuda":
        return None

    major, _ = torch.cuda.get_device_capability(device)

    if major < 7:
        return None

    if major >= 8:
        return torch.bfloat16
    
    return torch.float16


def autocast_context(device):
    dtype = get_amp_dtype(device)

    if dtype is None:
        return contextlib.nullcontext()

    return torch.autocast(device_type = "cuda", dtype = dtype)


def create_grad_scaler(device):
    enabled = get_amp_dtype(device) == torch.float16

    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled = enabled)

    return torch.cuda.amp.GradScaler(enabled = enabled)