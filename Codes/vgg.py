import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

import ssl


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        ssl._create_default_https_context = ssl._create_unverified_context
        vgg19_model = models.vgg19(pretrained=True)

        
        vgg19_model = vgg19_model.cuda()

        self.feature_extractor = self.get_vgg_feature_map(vgg19_model)

    def get_vgg_feature_map(self, vgg19_model):
        layers_dict = {
            'conv1_1': vgg19_model.features[0],
            'conv1_2': vgg19_model.features[2],
            'maxpool1': vgg19_model.features[4],
            'conv2_1': vgg19_model.features[5],
            'conv2_2': vgg19_model.features[7],
            'maxpool2': vgg19_model.features[9],
            'conv3_1': vgg19_model.features[10],
            'conv3_2': vgg19_model.features[12],
            'conv3_3': vgg19_model.features[14],
            'conv3_4': vgg19_model.features[16],
            'maxpool3': vgg19_model.features[18],
            'conv4_1': vgg19_model.features[19],
            'conv4_2': vgg19_model.features[21],
            'conv4_3': vgg19_model.features[23],
            'conv4_4': vgg19_model.features[25],
            'maxpool4': vgg19_model.features[27],
            'conv5_1': vgg19_model.features[28],
            'conv5_2': vgg19_model.features[30],
            'conv5_3': vgg19_model.features[32],
            'conv5_4': vgg19_model.features[34],
            'maxpool5': vgg19_model.features[36],
        }



        return layers_dict

    def forward(self, x):
        feature_outputs = {}
        for layer_name, layer in self.feature_extractor.items():
            x = layer(x)
            feature_outputs[layer_name] = x

        return feature_outputs
