# -*- coding: utf-8 -*-

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_model(model_name, out_features, in_channels, arch_params):
    preloaded_models = {"ResNet18": torchvision.models.resnet18}

    own_models = {"ConvNet": ConvNet, "MLP": MLP, "PureConvNet": PureConvNet, "CombResnet18": CombRenset18}

    if model_name in preloaded_models:
        model = preloaded_models[model_name](pretrained=False, num_classes=out_features, **arch_params)

        # Hacking ResNets to expect 'in_channels' input channel (and not three)
        del model.conv1
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif model_name in own_models:
        return own_models[model_name](out_features=out_features, in_channels=in_channels, **arch_params)
    else:
        raise ValueError(f"Model name {model_name} not recognized!")


def dim_after_conv2D(input_dim, stride, kernel_size):
    return (input_dim - kernel_size + 2) // stride


class CombRenset18(nn.Module):
    def __init__(self, out_features, in_channels):
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        #x = self.resnet_model.layer2(x)
        #x = self.resnet_model.layer3(x)
        #x = self.last_conv(x)
        x = self.pool(x)
        x = x.mean(dim=1)
        return x


class ConvNet(torch.nn.Module):
    def __init__(self, out_features, in_channels, kernel_size, stride, linear_layer_size, channels_1, channels_2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels_1, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=channels_1, out_channels=channels_2, kernel_size=kernel_size, stride=stride)

        output_shape = (4, 4)
        self.pool = nn.AdaptiveAvgPool2d(output_shape)

        self.fc1 = nn.Linear(in_features=output_shape[0] * output_shape[1] * channels_2, out_features=linear_layer_size)
        self.fc2 = nn.Linear(in_features=linear_layer_size, out_features=out_features)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, out_features, in_channels, hidden_layer_size):
        super().__init__()
        input_dim = in_channels * 40 * 20
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_layer_size)
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=out_features)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class PureConvNet(torch.nn.Module):

    act_funcs = {"relu": F.relu, "tanh": F.tanh, "identity": lambda x: x}

    def __init__(self, out_features, pooling, use_second_conv, kernel_size, in_channels,
                 channels_1=20, channels_2=20, act_func="relu"):
        super().__init__()
        self.use_second_conv = use_second_conv

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels_1, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv2d(in_channels=channels_1, out_channels=channels_2, kernel_size=kernel_size, stride=1)

        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        if pooling == "average":
            self.pool = nn.AdaptiveAvgPool2d(output_shape)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_shape)

        self.conv3 = nn.Conv2d(in_channels=channels_2 if use_second_conv else channels_1,
                               out_channels=1, kernel_size=1, stride=1)
        self.act_func = PureConvNet.act_funcs[act_func]

    def forward(self, x):
        x = self.act_func(self.conv1(x))
        if self.use_second_conv:
            x = self.act_func(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        return x
