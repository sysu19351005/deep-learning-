#计算ssim
import torch
import torch.nn.functional

from .utils import fspecial_gauss
from .utils import gaussian_filter

__all__ = [
    "ssim", "SSIM"
]


# Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
def ssim(source: torch.Tensor, target: torch.Tensor, filter_weight: torch.Tensor, cs=False) -> float:
    """Python implements structural similarity.

    Args:
        source (np.array): Original tensor picture.
        target (np.array): Target tensor picture.
        filter_weight (torch.Tensor): Gaussian filter weight.
        cs (bool): It is mainly used to calculate ms-ssim value. (default: False)

    Returns:
        Structural similarity value.
    """
    k1 = 0.01 ** 2
    k2 = 0.03 ** 2

    filter_weight = filter_weight.to(source.device)

    mu1 = gaussian_filter(source, filter_weight)
    mu2 = gaussian_filter(target, filter_weight)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(source * source, filter_weight) - mu1_sq
    sigma2_sq = gaussian_filter(target * target, filter_weight) - mu2_sq
    sigma12 = gaussian_filter(source * target, filter_weight) - mu1_mu2

    cs_map = (2 * sigma12 + k2) / (sigma1_sq + sigma2_sq + k2)
    cs_map = torch.nn.functional.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + k1) / (mu1_sq + mu2_sq + k1)) * cs_map
    out = ssim_map.mean([1, 2, 3])

    if cs:
        cs_out = cs_map.mean([1, 2, 3])
        return out, cs_out

    return out


class SSIM(torch.nn.Module):
    def __init__(self) -> None:
        super(SSIM, self).__init__()
        self.filter_weight = fspecial_gauss(11, 1.5)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source (torch.Tensor): Original tensor picture.
            target (torch.Tensor): Target tensor picture.

        Returns:
            torch.Tensor.
        """
        assert source.shape == target.shape
        out = torch.mean(ssim(source, target, filter_weight=self.filter_weight))

        return out
