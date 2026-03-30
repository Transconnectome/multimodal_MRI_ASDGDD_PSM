import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.densenetYA import DenseNetMM #model script

# setting path

class DenseNet_IntegratedGrad(DenseNetMM):
    def __init__(self, subject_data, args, growth_rate=32, block_config=(3, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, in_channels=1,
                 mode="encoder", memory_efficient=False, **kwargs):
        super().__init__(subject_data, args, growth_rate, block_config,
                         num_init_features, bn_size, drop_rate, num_classes,
                         in_channels, mode, memory_efficient,
                         **kwargs)
    
    def forward(self, imgs):
        outs = []
        results = {} 
        for i, extractor in enumerate(self.feature_extractors):
            if self.multimodal == 'multichannel':
                x = imgs.squeeze(2)
            elif self.multimodal == 'multifusion':
                x = imgs[:,i].squeeze(2) if i ==0 else imgs[:,i:].squeeze(2)
            else:
                x = imgs[:,i] # bring each modality
                
            features = extractor(x)
            if self.mode == "classifier":
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool3d(out, 1)
                out = torch.flatten(out, 1)
                outs.append(out)
                
        out = torch.cat(outs, 1)
        out = self.classifier[0](out).squeeze(dim=1)
            
        return out
    

def _densenet(arch, growth_rate, block_config, num_init_features, **kwargs):
    model = DenseNet_IntegratedGrad(growth_rate=growth_rate,block_config=block_config,
                       num_init_features=num_init_features, **kwargs)
    return model


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, **kwargs)