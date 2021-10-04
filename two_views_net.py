# Side Classifier: 2 views classifier using Patch Clf for CC, MLO

import sys
import collections
import torch
import torch.nn as nn

from bottleneck_ import Bottleneck
from single_full_net import SingleBreastClassifier

from EfficientNet_PyTorch.efficientnet_pytorch import EfficientNet, MBConvBlock

class TwoViewsMIDBreastClassifier(nn.Module):
    """
    Two-views:
    1-Take Feature Extractor part from Full image Classifier (which is the patch classifier part).
    2-Concatenate the outputs (activation maps) and output.
    Topology = MID. Means we concatenate feature extractor from each side and later will stack 
    top layers in the "middle of the way"
    """

    def __init__(self, network):
        super(TwoViewsMIDBreastClassifier, self).__init__()
        # self.single_clf_full = SingleBreastClassifier(device, MODEL_FULL_CLF)
        self.single_clf_full = SingleBreastClassifier(network)
        # take only the weights of 'patch classifier' part, disregard original Top layer(s)
        self.single_clf_core = self.single_clf_full.feature_extractor

    def forward(self, x):
        # x input comes in dict of views also
        x1_2 = self.single_clf_core(x[:, 0:3, :, :])
        x2_2 = self.single_clf_core(x[:, 3:6, :, :])
        hidden2 = torch.cat([x1_2, x2_2], dim=1)

        return hidden2


class SideMIDBreastModel(nn.Module):
    """ Calls TwoViewsMIDBreastClassifier that instantiates 2-views (CC+MLO) with
        concatenated output (without their original top layer).
        Then append the 2-views top layer, that can be resblocks or MBConv blocks,
        with 1, 2 or 0 count. The last means only FC layer.
        Strides: no. os strides for EficientNets
    """
    connections = 256
    output_size = (4, 2)

    def __init__(self, device, network, n_blocks, b_type='resnet', avg_pool=True, strides=1):
        super(SideMIDBreastModel, self).__init__()
        if n_blocks not in [0, 1, 2]:
            print('Wrong number of Top Layer blocks.')
            sys.exit()
        self.n_blocks = n_blocks
        self.two_views_clf = TwoViewsMIDBreastClassifier(network)
        self.avg_pool = avg_pool
        output_channels = 2048

        print('Creating Side Mid Networking using:', network, ' and Top Block type: ', b_type, ' Qty: ', n_blocks)

        if network == 'Resnet50':
            input_channels = 4096       # From concatenation
            if b_type == 'resnet':
                self.w_h = 9*7  # width and height of last layer output  
           
            if n_blocks == 1:
                self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
            elif n_blocks == 2:
                self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                self.block2 = Bottleneck(inplanes=output_channels, planes=512, stride=2)

        # elif network == 'EfficientNet-b4':
        elif 'EfficientNet' in network:
            input_channels = 3584       # From concatenation+

            if b_type == 'resnet':
                self.w_h = 9*7  # width and height of last layer output
                if n_blocks == 1:
                    self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                elif n_blocks == 2:
                    self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                    self.block2 = Bottleneck(inplanes=output_channels, planes=512, stride=2)
                else:
                    output_channels = 3584  # only FC

            elif b_type == 'mbconv':

                feat_ext = EfficientNet.from_name(network.lower(), num_classes=5)
                
                if 'b0' in network:
                    inplanes = 2560
                    output_channels=inplanes
                elif 'b4' in network:
                    inplanes = 3584
                    output_channels=inplanes

                self.w_h = 36*28                 # width and height of last layer output

                # Parameters for an individual model block
                BlockArgs = collections.namedtuple('BlockArgs', [
                    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
                    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

                new_block = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides], expand_ratio=2,
                                    input_filters=inplanes, output_filters=output_channels, se_ratio=0.25, id_skip=True)

                # below line for same params for blocks as main net
                block_args, global_params, image_size = new_block, feat_ext._global_params, [15, 15]

                if n_blocks == 1:
                    self.block1 = MBConvBlock(block_args, global_params, image_size=image_size)
                elif n_blocks == 2:
                    self.block1 = MBConvBlock(block_args, global_params, image_size=image_size)
                    new_block = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides], expand_ratio=2,
                                    input_filters=output_channels, output_filters=output_channels, se_ratio=0.25, id_skip=True)
                    self.block2 = MBConvBlock(new_block, global_params, image_size=image_size)

        if self.avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(output_channels, 2)

        else:
            # AVGPOOL
            self.fc_pre = nn.Linear(2048* self.w_h, 1024)  # for spatial features - Resnet 9*7 / EficientNet 36*28
            self.fc = nn.Linear(1024, 2)


    def forward(self, x):
        # x.shape: torch.Size([1, 3, 1152, 896])
        x = self.two_views_clf(x)       # out: torch.Size([1, 4096, 36, 28])
        if self.n_blocks == 1:
            x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
        elif self.n_blocks == 2:
            x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
            x = self.block2(x)          # out: Resnet([1, 2048, 9, 7]) / Eficientnet [2, 2048, 36, 28] 

        if self.avg_pool:
            x = self.avgpool(x)             # out: torch.Size([1, 2048, 1, 1])
            x = torch.flatten(x, 1)         # out: torch.Size([1, 2048])
        else:
            # NO AVGPOOL
            x = x.view(-1, 2048* self.w_h)
            x = self.fc_pre(x)

        x = self.fc(x)                  # out: torch.Size([1, 2]

        return x
