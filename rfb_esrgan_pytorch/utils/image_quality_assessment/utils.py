
import numpy as np
import torch
import torch.nn.functional

__all__ = [
    "gaussian_filter", "fspecial_gauss"
]


def gaussian_filter(x: torch.Tensor, filter_weight: torch.Tensor) -> torch.Tensor:
    r"""Gaussian filtering using two dimensional convolution.

    Args:
        x (torch.Tensor): Input tensor.
        filter_weight (torch.Tensor): Gaussian filter weight.

    Returns:
        torch.Tensor.
    """
    out = torch.nn.functional.conv2d(x, filter_weight, stride=1, padding=0, groups=x.shape[1])
    return out


def fspecial_gauss(size: int, sigma: float):
    r"""Implementation of Gaussian filter(MATLAB) in Python.

    Args:
        size (int): Wave filter size.
        sigma (float): Standard deviation of filter.

    Returns:
        Picture after using filter.
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    gauss = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    gauss = torch.from_numpy(gauss / gauss.sum()).float().unsqueeze(0).unsqueeze(0)
    out = gauss.repeat(3, 1, 1, 1)
    return out
