import torch
import torch.nn as nn

__all__ = [
    "ResidualDenseBlock", "ResidualInResidualDenseBlock",
    "ReceptiveFieldBlock", "ReceptiveFieldDenseBlock", "ResidualOfReceptiveFieldDenseBlock",
    "SubpixelConvolutionLayer"
]


class ResidualDenseBlock(nn.Module):
    r"""

    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.2)
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), dim=1))

        out = torch.add(conv5 * self.scale_ratio, x)

        return out


class ResidualInResidualDenseBlock(nn.Module):
    r"""

    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.2)
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels)

        self.scale_ratio = scale_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        out = torch.add(out * self.scale_ratio, x)

        return out


class ReceptiveFieldBlock(nn.Module):
    r"""

    Args:
        in_channels (int): Number of channels in the input image. (Default: 64)
        out_channels (int): Number of channels produced by the convolution. (Default: 64)
        scale_ratio (float): Residual channel scaling column. (Default: 0.1)
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, scale_ratio: float = 0.1):
        super(ReceptiveFieldBlock, self).__init__()
        branch_channels = in_channels // 4

        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 3, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels // 2, (branch_channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d((branch_channels // 4) * 3, branch_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        )

        self.conv_linear = nn.Conv2d(4 * branch_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv_linear(out)

        out = self.leaky_relu(torch.add(out * self.scale_ratio, shortcut))

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ReceptiveFieldDenseBlock(nn.Module):
    r""" Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
        RFB-SSD proposed Receptive Fields Block (RFB) for object detection

    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.1)
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.1):
        super(ReceptiveFieldDenseBlock, self).__init__()
        self.rfb1 = nn.Sequential(
            ReceptiveFieldBlock(channels + 0 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb2 = nn.Sequential(
            ReceptiveFieldBlock(channels + 1 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb3 = nn.Sequential(
            ReceptiveFieldBlock(channels + 2 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb4 = nn.Sequential(
            ReceptiveFieldBlock(channels + 3 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb5 = ReceptiveFieldBlock(channels + 4 * growth_channels, channels)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rfb1 = self.rfb1(x)
        rfb2 = self.rfb2(torch.cat((x, rfb1), dim=1))
        rfb3 = self.rfb3(torch.cat((x, rfb1, rfb2), dim=1))
        rfb4 = self.rfb4(torch.cat((x, rfb1, rfb2, rfb3), dim=1))
        rfb5 = self.rfb5(torch.cat((x, rfb1, rfb2, rfb3, rfb4), dim=1))

        out = torch.add(rfb5 * self.scale_ratio, x)

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    r"""The residual block structure of traditional RFB-ESRGAN is defined

    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.1)
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.1):
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.RFDB1 = ReceptiveFieldDenseBlock(channels, growth_channels)
        self.RFDB2 = ReceptiveFieldDenseBlock(channels, growth_channels)
        self.RFDB3 = ReceptiveFieldDenseBlock(channels, growth_channels)

        self.scale_ratio = scale_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RFDB1(x)
        out = self.RFDB1(out)
        out = self.RFDB1(out)

        out = torch.add(out * self.scale_ratio, x)

        return out


class SubpixelConvolutionLayer(nn.Module):
    r"""

    Args:
        channels (int): Number of channels in the input image. (Default: 64)
    """

    def __init__(self, channels: int = 64) -> None:
        super(SubpixelConvolutionLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.rfb1 = ReceptiveFieldBlock(channels, channels)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.rfb2 = ReceptiveFieldBlock(channels, channels)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample(x)
        out = self.rfb1(out)
        out = self.leaky_relu1(out)
        out = self.conv(out)
        out = self.pixel_shuffle(out)
        out = self.rfb2(out)
        out = self.leaky_relu2(out)

        return out
