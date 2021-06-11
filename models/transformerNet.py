import torch
from torch import nn


#Create convolution class for convenience, will be later use in the TransformerNet Model
class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        return self.conv2d(x)

#Create upsample class, will be later use in the TransformerNet Model

class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.upsampling_factor = stride
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv2d(x)

#Create Residual Block class, will be later use in the TransformerNet Model

class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        kernel_size = 3
        stride_size = 1
        self.conv1 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=stride_size)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=stride_size)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual  # modification: no ReLu after the addition


# the transformerNet model for image generation
# reference: https://arxiv.org/abs/1603.08155
# exact architecture: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        num_of_channels = [3, 32, 64, 128]
        kernel_sizes = [9, 3, 3]
        stride_sizes = [1, 2, 2]
        self.relu = nn.ReLU()
        self.conv1 = ConvLayer(num_of_channels[0], num_of_channels[1], kernel_sizes[0], stride_sizes[0])
        self.in1 = nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.conv2 = ConvLayer(num_of_channels[1], num_of_channels[2], kernel_sizes[1], stride_sizes[1])
        self.in2 = nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.conv3 = ConvLayer(num_of_channels[2], num_of_channels[3], kernel_sizes[2], stride_sizes[2])
        self.in3 = nn.InstanceNorm2d(num_of_channels[3], affine=True)

        self.res1 = ResidualBlock(num_of_channels[3])
        self.res2 = ResidualBlock(num_of_channels[3])
        self.res3 = ResidualBlock(num_of_channels[3])
        self.res4 = ResidualBlock(num_of_channels[3])
        self.res5 = ResidualBlock(num_of_channels[3])

        num_of_channels.reverse()
        kernel_sizes.reverse()
        stride_sizes.reverse()
        self.up1 = UpsampleConvLayer(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0],
                                     stride=stride_sizes[0])
        self.in4 = nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.up2 = UpsampleConvLayer(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1],
                                     stride=stride_sizes[1])
        self.in5 = nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.up3 = ConvLayer(num_of_channels[2], num_of_channels[3], kernel_size=kernel_sizes[2],
                             stride=stride_sizes[2])

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.up1(y)))
        y = self.relu(self.in5(self.up2(y)))

        return self.up3(y)