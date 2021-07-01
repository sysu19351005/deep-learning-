#计算psnr
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def tensor_to_y(image):
    """ Convert a BGR image to Y channels image. Reference source from https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        image (torch.Tensor): The input image is torch Tensor!

    Returns:
        Y channels tensor image.
    """
    tensor_to_image = cv2.cvtColor(np.asarray(transforms.ToPILImage()(image)), cv2.COLOR_RGB2BGR) / 255.
    y_image = np.dot(tensor_to_image, [24.966, 128.553, 65.481]) + 16.0
    tensor_y = transforms.ToTensor()(y_image.round())

    return tensor_y


class PSNR(torch.nn.Module):
    def __init__(self, gpu: int = None) -> None:
        super(PSNR, self).__init__()
        if gpu is None:
            self.mse_loss = nn.MSELoss().cuda(gpu)
        else:
            self.mse_loss = nn.MSELoss()

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source (torch.Tensor): Original tensor picture.
            target (torch.Tensor): Target tensor picture.

        Returns:
            torch.Tensor.
        """
        assert source.shape == target.shape
        out = 10 * torch.log10(1. / self.mse_loss(tensor_to_y(source), tensor_to_y(target)))

        return out

