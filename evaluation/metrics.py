import functools
import math

import torch
from skimage.metrics import structural_similarity as ssim

@average_on_batch
def SSIM(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    r"""
    Compute the SSIM between two input tensors x_pred and x_true. Both are assumed to be in the range [0, 1].
    """
    return ssim(
        x_pred[0, 0].detach().cpu().numpy(),
        x_true[0, 0].detach().cpu().numpy(),
        data_range=1,
    )


@average_on_batch
def PSNR(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    r"""
    Compute the PSNR between two input tensors x_pred and x_true. Both are assumed to be in the range [0, 1].
    """
    mse = torch.mean(torch.square(x_pred.flatten() - x_true.flatten()))
    if mse == 0:
        return 100
    return -20 * math.log10(math.sqrt(mse))
