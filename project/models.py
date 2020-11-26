# Adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models


# assume max pool with filter width 2 and stride 2
def get_width(input_width, kernel_size, pool):
    conv_width = (input_width - (kernel_size - 1))
    if pool:
        conv_width = conv_width // 2
    return conv_width


def compute_accuracy(predictions, y):
    assert len(predictions) == len(y), 'predictions dim does not match y dim'
    return np.mean(predictions == y)


def confusion_matrix(predictions, y, num_classes):
    assert len(predictions) == len(y), 'predictions dim does not match y dim'
    conf_mat = np.zeros((num_classes, num_classes))
    for i in range(len(y)):
        conf_mat[y[i], predictions[i]] += 1
    return conf_mat


def fp_fn(conf_mat, idx):
    FPs = np.sum(conf_mat[:, idx]) - conf_mat[idx, idx]
    FNs = np.sum(conf_mat[idx, :]) - conf_mat[idx, idx]
    TPs = conf_mat[idx, idx]
    TNs = np.sum(conf_mat) - (np.sum(conf_mat[:, idx]) + np.sum(conf_mat[idx, :]) - conf_mat[idx, idx])
    FPR = FPs / (FPs + TNs)
    TPR = TPs / (TPs + FNs)
    precision = TPs / (TPs + FPs)
    recall = TPs / (TPs + FNs)
    return {'FPR': FPR, 'TPR': TPR, 'precision': precision, 'recall': recall}


def eval_suite(predictions, y, label_names):
    conf_mat = confusion_matrix(predictions, y, len(label_names))
    return {'accuracy': compute_accuracy(predictions, y), 'confusion_matrix': conf_mat,
            'label_names': label_names, 'fp_fn': [fp_fn(conf_mat, idx) for idx in range(len(label_names))]}


class SmallModel(nn.Module):
    def __init__(self, num_classes, input_width, input_height, num_channels, num_layers=2, num_filters=[10, 20], kernel_sizes=[5, 5], pool=[True, True], dropout=[False,True]):
        super(SmallModel, self).__init__()
        assert len(num_filters) == num_layers, 'length of num_filters must match num_layers'
        assert len(kernel_sizes) == num_layers, 'length of kernel_sizes must match num_layers'
        assert len(pool) == num_layers, 'length of pool must match num_layers'
        assert len(dropout) == num_layers, 'length of dropout must match num_layers'
        self.num_classes = num_classes
        num_filters = [num_channels] + num_filters

        self.widths = [input_width]
        self.heights = [input_height]

        layers = []
        for layer in range(num_layers):
            layers.append(nn.Conv2d(num_filters[layer], num_filters[layer + 1], kernel_size=kernel_sizes[layer]))
            if dropout[layer]:
                layers.append(nn.Dropout2d())
            if pool[layer]:
                layers.append(nn.MaxPool2d(kernel_size=2))
            layers.append(nn.ReLU())

            self.widths.append(get_width(self.widths[-1], kernel_sizes[layer], pool[layer]))
            self.heights.append(get_width(self.heights[-1], kernel_sizes[layer], pool[layer]))
        self.convs = torch.nn.Sequential(*layers)

        self.ff_in_dim = self.widths[-1] * self.heights[-1] * num_filters[-1]
        self.fc1 = nn.Linear(self.ff_in_dim, self.ff_in_dim)
        self.fc2 = nn.Linear(self.ff_in_dim, num_classes)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.ff_in_dim)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


def set_parameter_requires_grad(model, feature_extracting):
    """Sets all layers to frozen or un-frozen

    Args:
        model (nn.Module): Model
        feature_extracting (bool): If true, all layers are frozen
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# TODO: currently input size has to be 224, may need to change this
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """Initialize Pretrained Model

    Args:
        model_name (str): Options: resnet, alexnet, vgg, squeezenet, densenet
        num_classes (int): Number of Classes
        feature_extract (bool): If True, all layers except last are frozen
        use_pretrained (bool, optional): Uses pretrained model if True. Defaults to True.

    Returns:
        nn.Module, int: Model, Input Size
    """    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
