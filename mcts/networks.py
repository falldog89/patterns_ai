""" Value and Policy Neural Networks defined in PyTorch for use in the MCTS algorithm:

State input is 8x8x102, which completely describes the current markov state.

Note that if we ever move to using a transformer, it may not be necessary to continue storing the state in this way

We create a CNN for the board, and MLPs for the other states.

We want to munge the conv net on the board state and the state vectors early on...

For now, we do all the CNN resnet stuff, followed by the MLP fully connected layers
to blend the state with the image.
"""
import torch
from torch import nn
import math


class PatternsNet(nn.Module):
    """ Take in the initial munging of the board and player states, and perform standard resnet
    fun on it:
    """
    def __init__(self,
                 in_channels: int = 102,
                 out_channels: int = 64,
                 ) -> None:
        """ 102 in channels for patterns. 18 for color and player, 72 for color group order, 12 for bowl tokens.
         """
        super(PatternsNet, self).__init__()

        # initial processing layer: output size (samples, out_channels, 8, 8)
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Downsampling layers: output size (samples, out * 2, 4, 4)
        self.down1 = ResidualLayer(in_channels=out_channels, stride=2)
        out_channels *= 2
        # (samples, out * 4, 2, 2)
        self.down2 = ResidualLayer(in_channels=out_channels, stride=2)
        out_channels *= 2

        # Standard resnet layers:
        # (samples, out * 4, 2, 2)
        self.standard1 = ResidualLayer(in_channels=out_channels, stride=1)
        self.standard2 = ResidualLayer(in_channels=out_channels, stride=1)

        # different heads:
        self.twoheadlayer = TwoHeadNet(in_channels=out_channels,
                                       value_out_channels=out_channels // 16,
                                       policy_out_channels=out_channels // 4,
                                       action_space=(106,))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ pass the input through the initial conv layer, expanding input channels to the number
        of filters required, before passing through the residual blocks """
        x = self.input_layer(x)
        x = self.down2(self.down1(x))
        x = self.standard2(self.standard1(x))

        return self.twoheadlayer(x)


class TwoHeadNet(nn.Module):
    """ Two different heads, attached to resnet backbone:

    One head addresses the policy prediction (with action-space number of neurons)
    one head addresses the value prediction (with a single neuron output, scaled to -1, 1
    """
    def __init__(self,
                 in_channels: int,
                 value_out_channels: int,
                 policy_out_channels: int,
                 action_space: tuple = (106,)) -> None:
        super(TwoHeadNet, self).__init__()
        self.action_space = action_space

        # output is float between -1 and 1 estimating board state
        self.value_head = nn.Sequential(
            # final bespoke convolutional layer for value:
            nn.Conv2d(in_channels, value_out_channels, kernel_size=3, padding=1, stride=1),# bias=False),
            nn.BatchNorm2d(value_out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # squash to flat:
            nn.Flatten(),
            nn.Linear(in_features=value_out_channels, out_features=1),
            # squish to between -1 and 1 to estimate the result of the game:
            nn.Tanh(),
        )

        # output is logits for softmax to give distribution over action space
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels, policy_out_channels, kernel_size=3, padding=1, stride=1),# bias=False),
            nn.BatchNorm2d(policy_out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=policy_out_channels, out_features=math.prod(action_space)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ produce both outputs """
        value_x = self.value_head(x)
        policy_x = self.policy_head(x).view((-1, *self.action_space))

        return value_x, policy_x


class ResidualLayer(nn.Module):
    """ Stick two residual basic blocks together, apply down-sampling as required:
    """
    def __init__(self,
                 in_channels: int = 128,
                 stride: int = 1) -> None:
        super(ResidualLayer, self).__init__()

        # The first block can be a down-sampling layer:
        self.basic_block_1 = ResidualBlock(in_channels=in_channels, stride=stride)
        self.basic_block_2 = ResidualBlock(in_channels=in_channels * stride, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ basic_block_1 will return x + conv(x)
        """
        x1 = self.basic_block_1(x)
        return self.basic_block_2(x1)


class ResidualBlock(nn.Module):
    """ A residual block of a resnet.

    Note that channels in and out are assumed equal, to allow for addition with manipulation.
    """
    def __init__(self,
                 in_channels: int,
                 stride: int = 1) -> None:
        super(ResidualBlock, self).__init__()

        self.is_downsample = True if stride != 1 else False

        out_channels = stride * in_channels

        # first layer can be downsample:
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # if downsampling, also downsample the residual for consistent tensor size:
        if self.is_downsample:
            # skip block needs a 1x1 conv to increase in channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ conv block applied to x, with skip connection to allow for robust residual training """
        residual = x if not self.is_downsample else self.downsample(x)
        return self.relu(residual + self.conv_2(self.conv_1(x)))
