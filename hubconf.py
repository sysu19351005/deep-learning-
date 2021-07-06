
import torch
from torch.hub import load_state_dict_from_url

from rfb_esrgan_pytorch.models.generator import Generator

model_urls = {
    "rfb_esrgan": None
}

dependencies = ["torch"]


def _gan(arch: str, pretrained: bool, progress: bool) -> Generator:
    model = Generator()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def rfb_esrgan(pretrained: bool = False, progress: bool = True) -> Generator:

    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("rfb_esrgan", pretrained, progress)
