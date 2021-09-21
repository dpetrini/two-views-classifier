
# Resnet blocks for network creation.
# To be improved

import torch
from torch import nn

from bottleneck_ import Bottleneck


class ResnetBlocks(nn.Module):
    def __init__(self, **kwargs):
        super(ResnetBlocks, self).__init__()
        inplanes = kwargs.get('inplanes') if kwargs.get('inplanes') else 2048
        self.block1 = Bottleneck(inplanes=inplanes, planes=512, stride=2)
        self.block2 = Bottleneck(inplanes=2048, planes=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # parece que Sinai eh 7x7
        self.fc = nn.Linear(2048, 2)

        # Change default init conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                # For mode=fan_in, the variance of the distribution is
                #  ensured in the forward pass, while for mode=fan_out,
                #  it is ensured in the backwards pass.
                if m.bias:              # Novo 2020-08-18- testar
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init.normal(m.weight, std=1e-3)
            #     if m.bias:
            #         init.constant(m.bias, 0)

    # Reimplementing forward pass
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Resnet4Blocks(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet4Blocks, self).__init__()
        inplanes = kwargs.get('inplanes') if kwargs.get('inplanes') else 2048
        self.block1 = Bottleneck(inplanes=inplanes, planes=512, stride=2)
        self.block1_1 = Bottleneck(inplanes=2048, planes=512, stride=1)
        self.block2 = Bottleneck(inplanes=2048, planes=512, stride=2)
        self.block2_2 = Bottleneck(inplanes=2048, planes=512, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

        # Change default conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block1_1(x)
        x = self.block2(x)
        x = self.block2_2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Resnet1Blocks(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet1Blocks, self).__init__()
        inplanes = kwargs.get('inplanes') if kwargs.get('inplanes') else 2048
        self.block1 = Bottleneck(inplanes=inplanes, planes=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

        # Change default init conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Resnet0Blocks(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet0Blocks, self).__init__()
        inplanes = kwargs.get('inplanes') if kwargs.get('inplanes') else 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes, 2)

        # Change default init conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
