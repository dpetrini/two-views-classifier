# Model for single image classifier
import sys
import collections
import torch
import torch.nn as nn

from resnet50 import MyResnet50
from resnet_blocks import ResnetBlocks, Resnet1Blocks

from EfficientNet_PyTorch.efficientnet_pytorch import EfficientNet, MBConvBlock


class EFBlocks(nn.Module):
    """ EfficientNet Top Block """
    def __init__(self, global_params, image_size=None, inplanes=1792, outplanes=2048, n_blocks=1, strides=1, **kwargs):
        super(EFBlocks, self).__init__()

        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon
        self.drop_connect_rate = global_params.drop_connect_rate
        self.n_blocks = n_blocks

        # Parameters for an individual model block
        BlockArgs = collections.namedtuple('BlockArgs', [
            'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
            'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

        block_args = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides],
                               expand_ratio=2, input_filters=inplanes,
                               output_filters=outplanes, se_ratio=0.25, id_skip=True)

        self._blocks = nn.ModuleList([])
        for _ in range(0, n_blocks):
            self._blocks.append(MBConvBlock(block_args, global_params, image_size=image_size))

        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        # self._conv_head = Conv2d(outplanes, outplanes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(num_features=outplanes, momentum=bn_mom, eps=bn_eps)
        # self.swish = MemoryEfficientSwish()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(global_params.dropout_rate/2)
        self.fc = nn.Linear(outplanes, 2)

    def forward(self, x):
        for i, block in enumerate(self._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(i) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate)

        # x = self._conv_head(x)
        # x = self.bn1(x)
        # x = self.swish(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class SingleBreastClassifier(nn.Module):
    """
    SingleBreast Classifier is the Full image classifier.
    In this class we reproduce the single-view model to separate in two parts:
    Feature extractor => that is the patch classifier
    Top Layer => the top blocks. [As for the AACR abstracts no. of blocks=1 ]
    Usually we will use only feature extractor for 2-views classifier.
    """

    def __init__(self, network):
        super(SingleBreastClassifier, self).__init__()


        if network == 'Resnet50':
            # instanciate resnet with 5 outputs
            patch_clf = MyResnet50(num_classes=5)

            # Join Models
            self.feature_extractor = nn.Sequential(
                *list(patch_clf.children())[:-2],     # remove FC layer
            )

            self.top_layer = Resnet1Blocks()

        # elif network == 'EfficientNet-b4':
        elif 'EfficientNet' in network:    
            patch_clf_model = EfficientNet.from_name(network.lower(), num_classes=5)  # Chenged default to exclude Top in EfficientNet

            if 'b0' in network:
                output_filters = 1280
                inplanes = 1280
            elif 'b4' in network:
                output_filters = 1792  #2048
                inplanes = 1792

            self.feature_extractor = patch_clf_model

            # Parameters for an individual model block
            BlockArgs = collections.namedtuple('BlockArgs', [
                'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
                'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

            new_block = BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=2,
                                  input_filters=inplanes, output_filters=output_filters, se_ratio=0.25, id_skip=True) # 1 block

            self.top_layer = EFBlocks(patch_clf_model._global_params, [15, 15], inplanes=inplanes,
                             outplanes=output_filters, n_blocks=2, strides=2)

        else:
            print('Wrong selection of network')


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.top_layer(x)

        return x
