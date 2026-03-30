import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()        
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""3D-DenseNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        mode (str) - specify in which mode DenseNet is trained on -- must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(3, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, in_channels=1,
                 mode="encoder", memory_efficient=False):

        super(DenseNet, self).__init__()

        assert mode in {'encoder', 'classifier', 'correct_encoder'}, "Unknown mode selected: %s"%mode

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.mode = mode
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_features = num_features

        if self.mode == "classifier":
            # Final batch norm
            self.features.add_module('norm5', nn.BatchNorm3d(num_features))
            # Linear layer
            self.classifier = nn.Linear(num_features, num_classes)
        elif self.mode == "encoder":
            self.hidden_representation = nn.Linear(num_features, 512)
            self.head_projection = nn.Linear(512, 128)
        elif self.mode == "correct_encoder": #TRY fix : BN inside model 
            self.features.add_module('norm5', nn.BatchNorm3d(num_features))
            self.hidden_representation = nn.Linear(num_features, 512)
            self.head_projection = nn.Linear(512, 128)

        # Init. with kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ## Eventually keep the input images for visualization
        self.input_imgs = x.detach().cpu().numpy()
        features = self.features(x)
        if self.mode == "classifier":
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool3d(out, 1)
            out = torch.flatten(out, 1)
            out = self.classifier(out)
        elif self.mode == "encoder":
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool3d(out, 1)
            out = torch.flatten(out, 1)

            out = self.hidden_representation(out)
            out = F.relu(out, inplace=True)
            out = self.head_projection(out)
        elif self.mode == "correct_encoder": #TRY fix : BN inside model 
            out = F.relu(features, inplace = True)
            out = F.adaptive_avg_pool3d(out , 1)
            out = torch.flatten(out , 1)
            
            out = self.hidden_representation(out)
            out = F.relu(out, inplace = True)
            out = self.head_projection(out)
        
    
        return out.squeeze(dim=1)

    def get_current_visuals(self):
        return self.input_imgs

    
class DenseNetMM(nn.Module):
    r"""3D-DenseNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Modified for the multimodal learning
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        mode (str) - specify in which mode DenseNet is trained on -- must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, subject_data, args, growth_rate=32, block_config=(3, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, in_channels=1,
                 mode="encoder", memory_efficient=False):

        super(DenseNetMM, self).__init__()

        assert mode in {'encoder', 'classifier', 'correct_encoder'}, "Unknown mode selected: %s"%mode
        self.mode = mode
        self.targets = args.cat_target + args.num_target
        self.multimodal = args.multimodal
        if hasattr(args,'dropout'):
            drop_rate = args.dropout
        self.drop_rate = drop_rate
        # self.out_dim = subject_data[self.targets[0]].nunique()
        
        self.feature_extractors = nn.ModuleList()
        if self.multimodal == 'multichannel':
            len_extractors = 1
            modalities = ['multichannel']
            in_channels_list = [len(args.data_type)]
        elif self.multimodal == 'multifusion':
            len_extractors = 2
            modalities = ['T1', 'DTI']
            in_channels_list = [1, len(args.data_type)-1]
        else:
            len_extractors = len(args.data_type)
            modalities = list(map(lambda x: x.split("_")[0], args.data_type))
            in_channels_list = [in_channels]*len_extractors
            
        for i in range(len_extractors):
            module_name = modalities[i]
            in_channels = in_channels_list[i]
            features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))
            num_features = num_init_features
            
            for i, num_layers in enumerate(block_config):
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    memory_efficient=memory_efficient
                )
                features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                    features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
            features.add_module('norm5', nn.BatchNorm3d(num_features))
            self.num_features = num_features
            
            self.feature_extractors.add_module(module_name, features)

        if self.mode == "classifier":
            # Linear layer
            FClayer = []

            for cat_label in args.cat_target:
                out_dim = subject_data[cat_label].nunique()                       
                FClayer.append(nn.Sequential(nn.Linear(num_features*len_extractors, out_dim)))

            for num_label in args.num_target:
                FClayer.append(nn.Sequential(nn.Linear(num_features*len_extractors, 1)))


            self.classifier = nn.ModuleList(FClayer)
            # suppose that we are going to concatenate output of each feature extractor

        elif self.mode == "correct_encoder": #TRY fix : BN inside model 
            self.hidden_representation = nn.Linear(num_features*len(args.data_type), 512)
            self.head_projection = nn.Linear(512, 128)

        # Init. with kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        ## forward
        outs = []
        results = {} 
        # shape of imgs == Batch_size(B) x Num_modalities(M) x 1(num_channel) x D x H x W (depth, height, width)
        for i, extractor in enumerate(self.feature_extractors): # forward for each feature_extractor
            if self.multimodal == 'multichannel':
                x = imgs.squeeze(2) # B x M(==Channel) x D x H x W
            elif self.multimodal == 'multifusion':
                x = imgs[:,i].squeeze(2) if i ==0 else imgs[:,i:].squeeze(2)
                # B x M(1 for T1, M-1 for DTI) x D x H x W
            else:
                x = imgs[:,i] # bring each modality, B x 1 x D x H x W
                
            features = extractor(x) # B x 1024(Hidden_dim, H) x D' x H' x W'
            
            if self.mode == "classifier":
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool3d(out, 1)
                out = torch.flatten(out, 1) # B x 1024                   
                outs.append(out)

            elif self.mode == "correct_encoder":
                out = F.relu(features, inplace = True)
                out = F.adaptive_avg_pool3d(out , 1)
                out = torch.flatten(out , 1)

                out = self.hidden_representation(out)
                out = F.relu(out, inplace = True)
                out = self.head_projection(out)
                outs.append(out)
                if len(outs) == len(self.feature_extractors):
                    return torch.cat(outs, 1).squeeze(dim=1)
                
        out = torch.cat(outs, 1) # concatenate multimodal features, B x MH
        for i, target in enumerate(self.targets):
            results[target] = self.classifier[i](out).squeeze(dim=1)
        results['embeddings'] = torch.stack(outs, 1).detach().cpu() # keep multimodal embedding shape, B x M x H
            
        return results

def _densenet(arch, growth_rate, block_config, num_init_features, **kwargs):
    model = DenseNetMM(growth_rate=growth_rate,block_config=block_config,
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
