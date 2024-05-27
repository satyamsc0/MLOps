
# import all necessary libraries/modules/packages
import torch
import torch.nn as nn

from collections import OrderedDict

# function for creating a convolution block
def conv_block(in_channels, out_channels, pool=False):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool:
    layers.append(nn.MaxPool2d(2))

  return nn.Sequential(*layers)

# Class for creating a Resnet model
class ResnetX(nn.Module):
  def __init__(self, in_channels, num_classes):
    super().__init__()

    self.conv1 = conv_block(in_channels, 64)
    self.conv2 = conv_block(64, 128, pool=True)
    self.res1 = nn.Sequential(OrderedDict([("conv1res1", conv_block(128, 128)), ("conv2res2", conv_block(128, 128))]))
    # used ordered dict just to show the name of the layer instead of the number

    self.conv3 = conv_block(128, 256, pool=True)
    self.conv4 = conv_block(256, 512, pool=True)
    self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

    self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                    nn.Flatten(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, num_classes))

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.res1(out) + out
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.res2(out) + out
    return self.classifier(out)

if __name__ == "__main__":
    model = ResnetX(3, 10)  # creating instance of the model ResnetX
    print(model)             # printing the summary of the model