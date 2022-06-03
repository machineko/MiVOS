# Credit: https://github.com/VainF/DeepLabV3Plus-Pytorch
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from model.s2m.utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]

from pathlib import Path

import torch
from torch import nn
import numpy as np
import ujson
from torch.nn import functional as F

class SimpleForward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

# with torch.no_grad():
#     # simplelinear = nn.Linear(in_features=10, out_features=20, dtype=torch.float32, bias=False)
#     simplelinear2 = nn.Linear(in_features=10, out_features=20, dtype=torch.float32, bias=True)
#
#     np.save("linear2.npy", simplelinear2.weight.numpy())
#     np.save("linear2bias.npy", simplelinear2.bias.numpy())
#     print(simplelinear2)
# linear = np.load("linear2.npy")
# zz = nn.Linear(in_features=10, out_features=20, bias=True)
# zz.weight = torch.nn.Parameter(torch.from_numpy(linear))
# zz.bias = torch.nn.Parameter(torch.from_numpy(np.load("linear2bias.npy")))
# print(linear)
# out = zz.forward(torch.ones(10, 20, 10))
# print(zz.forward(torch.ones(10, 20, 10)))
# print(out[0, ...])
# s2m = torch.load("s2m.pth")
# norm = torch.load("bn1.pth")
# norm.eval()
# print(norm)
# ze = nn.Linear(in_features=10, out_features=20, bias=False)
#
# def generate_path(path: str, name: str) -> str:
#     path = Path(path).resolve()
#     path.mkdir(exist_ok=True, parents=True)
#     path = Path(path, name)
#     save_path = str(path)
#     return save_path
#
# def save_linear(layer: nn.Linear, path: str, name: str):
#     # TODO FIX
#     save_path = generate_path(path, name)
#     np.save(save_path + "_weight.npy", layer.weight.numpy())
#     #params["layer_type"] = "linear"
#
#     if layer.bias:
#         np.save(save_path + "_bias.npy", layer.weight.numpy())
#
# def save_batchnorm2d(layer: nn.BatchNorm2d, path: str, name: str, idx: int):
#     save_path = generate_path(path, name + f"_{idx}")
#     keys = ["eps", "momentum"]
#     params = {}
#     layer_dict = layer.__dict__
#     params["layer_type"] = "batchNorm2d"
#     params["name"] = name
#     params["idx"] = idx
#     for k in keys:
#         params[f"{k}"] = layer_dict[k]
#     params["weightsPath"] = save_path + "_weights.npy"
#     params["biasPath"] = save_path + "_bias.npy"
#     params["num_batches_tracked"] = save_path + "_num_batches_tracked.npy"
#     params["running_mean"] = save_path + "_running_mean.npy"
#     params["running_var"] = save_path + "_running_var.npy"
#     np.save(save_path + "_weights.npy", layer.weight.numpy()[None, :, None, None])
#     np.save(save_path + "_bias.npy", layer.bias.numpy()[None, :, None, None])
#     np.save(save_path + "_num_batches_tracked.npy", layer.num_batches_tracked.numpy().astype(np.float32))
#     np.save(save_path + "_running_mean.npy", layer.running_mean.numpy()[None, :, None, None])
#     np.save(save_path + "_running_var.npy", layer.running_var.numpy())
#     return params
#
# def save_conv2d(layer: nn.Conv2d, path: str, name: str, idx: int) -> dict:
#     save_path = generate_path(path, name + f"_{idx}")
#     keys = ["dilation2d", "kernel_size2d", "padding2d", "stride2d", "in_channels","groups", "out_channels"]
#     params = {}
#     layer_dict = layer.__dict__
#     params["name"] = name
#     params["idx"] = idx
#     params["paddingMode"] = layer.padding_mode
#     params["layer_type"] = "conv2d"
#     for k in keys:
#         tmp_k = k.replace("2d", "")
#         if "2d" not in k:
#             params[f"{k}"] = layer_dict[k]
#         else:
#             if not isinstance(layer_dict[tmp_k], (list, tuple)):
#                 params[f"{k}"] = [layer_dict[tmp_k], layer_dict[tmp_k]]
#             else:
#                 params[f"{k}"] = layer_dict[tmp_k]
#
#     if isinstance(layer.bias,  torch.Tensor):
#         params["useBias"] = True
#         params["biasPath"] = save_path + "_bias.npy"
#         np.save(params["biasPath"], layer.bias)
#     else:
#         params["useBias"] = False
#     params["weightsPath"] = save_path + "_weights.npy"
#     params["dataLayout"] = "NCHW"
#     np.save(params["weightsPath"], layer.weight)
#     return params
#
# def save_maxpool2d(layer: nn.MaxPool2d, name: str, idx: int) -> dict:
#     keys = ["dilation2d", "kernel_size2d", "padding2d", "stride2d"]
#     params = {}
#     params["layer_type"] = "maxPool2d"
#     layer_dict = layer.__dict__
#     params["data_layout"] = "NCHW"
#     params["name"] = name
#     params["idx"] = idx
#     params["paddingMode"] = "zeros"
#     for k in keys:
#         tmp_k = k.replace("2d", "")
#         if "2d" not in k:
#             params[f"{k}"] = layer_dict[k]
#         else:
#             if not isinstance(layer_dict[tmp_k], (list, tuple)):
#                 params[f"{k}"] = [layer_dict[tmp_k], layer_dict[tmp_k]]
#             else:
#                 params[f"{k}"] = layer_dict[tmp_k]
#
#     params["ceil_mode"] = layer_dict["ceil_mode"]
#     return params
#
# def save_adaptiveavgpool2d(layer: nn.AdaptiveAvgPool2d, name: str, idx: int) -> dict:
#     params = {}
#     params["layer_type"] = "adaptiveAvgPool2d"
#     params["data_layout"] = "NCHW"
#     params["name"] = name
#     params["idx"] = idx
#     params["outSize2d"] = []
#     if isinstance(layer.output_size, tuple):
#         for i in layer.output_size:
#             if i:
#                 params["outSize2d"].append(i)
#             else:
#                 params["outSize2d"].append(-1)
#     else:
#         params["outSize2d"] = [layer.output_size, layer.output_size]
#     return params
#
# def save_relu(layer: nn.ReLU, name: str, idx: int) -> dict:
#     params = {}
#     params["layer_type"] = "relu"
#     params["name"] = name
#     params["idx"] = idx
#     params["relu_inplace"] = layer.inplace
#     return params
#
# def save_dropout(layer: nn.Dropout, name: str, idx: int) -> dict:
#     params = {}
#     params["layer_type"] = "dropout"
#     params["name"] = name
#     params["idx"] = idx
#     params["probability"] = layer.p
#     return params
#
# @dataclass
# class Parser:
#     path: str
#     data: dict = field(default_factory=dict)
#     idx: int = 0
#
#     def parse_module(self, module, name: str):
#
#         if isinstance(module, (nn.ModuleDict, nn.ModuleList, nn.Sequential)):
#             for (new_name, new_module) in module.named_children():
#                 self.parse_module(new_module, name=f"{name}.{new_name}")
#             return
#         elif list(module.children()) != []:
#             module_name = module.__class__.__name__
#             print(module_name, "Not supported")
#         elif isinstance(module, nn.ReLU):
#             data = save_relu(module, name=name, idx=self.idx)
#         elif isinstance(module, nn.AdaptiveAvgPool2d):
#             data = save_adaptiveavgpool2d(module, name=name, idx=self.idx)
#         elif isinstance(module, nn.Conv2d):
#             data = save_conv2d(module, name=name, path=self.path, idx=self.idx)
#         elif isinstance(module, nn.BatchNorm2d):
#             data = save_batchnorm2d(module, name=name, path=self.path, idx=self.idx)
#         elif isinstance(module, nn.Dropout):
#             data = save_dropout(module, name=name, idx=self.idx)
#         elif isinstance(module, nn.MaxPool2d):
#             data = save_maxpool2d(module, name=name, idx=self.idx)
#         else:
#             raise NotImplementedError(type(module))
#         self.idx += 1
#         self.data["layers"].append(data)
import time
# with torch.no_grad():
#     conv = torch.load("conv2d.pth")
#     # conv.eval()
#     batch = torch.load("bn1.pth")
#     batch.eval()
#     maxpool = torch.load("maxpool.pth")
#     maxpool2 = nn.MaxPool2d(kernel_size=3, ceil_mode=False, dilation=1, padding=1,
#                            return_indices=False, stride=2)
#     cc = conv(torch.ones(1,6,128,128))
#     avgpool1 = nn.AdaptiveAvgPool2d(1)
#     avgpool2 = nn.AvgPool2d(kernel_size=(1,1))
#
#
#     all_data = {"layers": []}
#     # if isinstance(conv, nn.Conv2d):
#     all_data["layers"].append(save_conv2d(conv, "inter_conv", "conv", 1))
#     all_data["layers"].append(save_conv2d(conv, "inter_conv", "conv", 2))
#     all_data["layers"].append(save_maxpool2d(maxpool, "maxpool", 3))
#     all_data["layers"].append(save_adaptiveavgpool2d(avgpool1, "avg2d", 4))
#     all_data["layers"].append(save_batchnorm2d(batch, "inter_conv", "bn", 13))
#     with open("elo.json", "w") as f:
#         ujson.dump(all_data, f)

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self.is_first = True
        self._init_weight()

    def forward(self, feature):
        if self.is_first:
            #self.save_all()
            self.is_first = False
            # return
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=True)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )

    @torch.no_grad()
    def save_all(self):
        all_proj = {"layers": []}
        idx = 0
        for (name, module) in self.project.named_modules():
            parsed = parse_module(module, idx=idx, name=name, path="proj_data")
            if parsed:
                all_proj["layers"].append(parsed)
        with open("project.json", "w") as f:
            ujson.dump(all_proj, f)
        print("Xd")
        return

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        out = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.isfirst = True
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        if self.isfirst:
            self.export_all()
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

    def export_all(self):
        seq1 = {"layers": []}
        conv1 = {"layers": []}
        conv2 = {"layers": []}
        conv3 = {"layers": []}
        pool1 = {"layers": []}
        project = {"layers": []}
        parser = Parser(path="seq1", data=seq1)
        for (name, module) in self.convs[0].named_children():
            parser.parse_module(module, name=name)
        with open("seq1.json", "w") as f:
            ujson.dump(parser.data, f)

        parser = Parser(path="conv1", data=conv1)
        for (name, module) in self.convs[1].named_children():
            parser.parse_module(module, name=name)
        with open("conv1.json", "w") as f:
            ujson.dump(parser.data, f)

        parser = Parser(path="conv2", data=conv2)
        for (name, module) in self.convs[2].named_children():
            parser.parse_module(module, name=name)
        with open("conv2.json", "w") as f:
            ujson.dump(parser.data, f)

        parser = Parser(path="conv3", data=conv3)
        for (name, module) in self.convs[3].named_children():
            parser.parse_module(module, name=name)
        with open("conv3.json", "w") as f:
            ujson.dump(parser.data, f)

        parser = Parser(path="pool1", data=pool1)
        for (name, module) in self.convs[4].named_children():
            parser.parse_module(module, name=name)
        with open("pool1.json", "w") as f:
            ujson.dump(parser.data, f)

        parser = Parser(path="project", data=project)
        for (name, module) in self.convs[4].named_children():
            parser.parse_module(module, name=name)
        with open("project.json", "w") as f:
            ujson.dump(parser.data, f)
        return

def generate_path(path: str, name: str) -> str:
    path = Path(path).resolve()
    path.mkdir(exist_ok=True, parents=True)
    path = Path(path, name)
    save_path = str(path)
    return save_path

def save_linear(layer: nn.Linear, path: str, name: str):
    # TODO FIX
    save_path = generate_path(path, name)
    np.save(save_path + "_weight.npy", layer.weight.numpy())
    #params["layer_type"] = "linear"

    if layer.bias:
        np.save(save_path + "_bias.npy", layer.weight.numpy())

def save_batchnorm2d(layer: nn.BatchNorm2d, path: str, name: str, idx: int):
    save_path = generate_path(path, name + f"_{idx}")
    keys = ["eps", "momentum"]
    params = {}
    layer_dict = layer.__dict__
    params["layer_type"] = "batchNorm2d"
    params["name"] = name
    params["idx"] = idx
    for k in keys:
        params[f"{k}"] = layer_dict[k]
    params["weightsPath"] = save_path + "_weights.npy"
    params["biasPath"] = save_path + "_bias.npy"
    params["num_batches_tracked"] = save_path + "_num_batches_tracked.npy"
    params["running_mean"] = save_path + "_running_mean.npy"
    params["running_var"] = save_path + "_running_var.npy"
    np.save(save_path + "_weights.npy", layer.weight.numpy()[None, :, None, None])
    np.save(save_path + "_bias.npy", layer.bias.numpy()[None, :, None, None])
    np.save(save_path + "_num_batches_tracked.npy", layer.num_batches_tracked.numpy().astype(np.float32))
    np.save(save_path + "_running_mean.npy", layer.running_mean.numpy()[None, :, None, None])
    np.save(save_path + "_running_var.npy", layer.running_var.numpy()[None, :, None, None])
    return params

def save_conv2d(layer: nn.Conv2d, path: str, name: str, idx: int) -> dict:
    save_path = generate_path(path, name + f"_{idx}")
    keys = ["dilation2d", "kernel_size2d", "padding2d", "stride2d", "in_channels","groups", "out_channels"]
    params = {}
    layer_dict = layer.__dict__
    params["name"] = name
    params["idx"] = idx
    params["paddingMode"] = layer.padding_mode
    params["layer_type"] = "conv2d"
    for k in keys:
        tmp_k = k.replace("2d", "")
        if "2d" not in k:
            params[f"{k}"] = layer_dict[k]
        else:
            if not isinstance(layer_dict[tmp_k], (list, tuple)):
                params[f"{k}"] = [layer_dict[tmp_k], layer_dict[tmp_k]]
            else:
                params[f"{k}"] = layer_dict[tmp_k]

    if isinstance(layer.bias,  torch.Tensor):
        params["useBias"] = True
        params["biasPath"] = save_path + "_bias.npy"
        np.save(params["biasPath"], layer.bias)
    else:
        params["useBias"] = False
    params["weightsPath"] = save_path + "_weights.npy"
    params["dataLayout"] = "NCHW"
    np.save(params["weightsPath"], layer.weight)
    return params

def save_maxpool2d(layer: nn.MaxPool2d, name: str, idx: int) -> dict:
    keys = ["dilation2d", "kernel_size2d", "padding2d", "stride2d"]
    params = {}
    params["layer_type"] = "maxPool2d"
    layer_dict = layer.__dict__
    params["data_layout"] = "NCHW"
    params["name"] = name
    params["idx"] = idx
    params["paddingMode"] = "zeros"
    for k in keys:
        tmp_k = k.replace("2d", "")
        if "2d" not in k:
            params[f"{k}"] = layer_dict[k]
        else:
            if not isinstance(layer_dict[tmp_k], (list, tuple)):
                params[f"{k}"] = [layer_dict[tmp_k], layer_dict[tmp_k]]
            else:
                params[f"{k}"] = layer_dict[tmp_k]

    params["ceil_mode"] = layer_dict["ceil_mode"]
    return params

def save_adaptiveavgpool2d(layer: nn.AdaptiveAvgPool2d, name: str, idx: int) -> dict:
    params = {}
    params["layer_type"] = "adaptiveAvgPool2d"
    params["data_layout"] = "NCHW"
    params["name"] = name
    params["idx"] = idx
    params["outSize2d"] = []
    if isinstance(layer.output_size, tuple):
        for i in layer.output_size:
            if i:
                params["outSize2d"].append(i)
            else:
                params["outSize2d"].append(-1)
    else:
        params["outSize2d"] = [layer.output_size, layer.output_size]
    return params

def save_relu(layer: nn.ReLU, name: str, idx: int) -> dict:
    params = {}
    params["layer_type"] = "relu"
    params["name"] = name
    params["idx"] = idx
    params["relu_inplace"] = layer.inplace
    return params

def save_dropout(layer: nn.Dropout, name: str, idx: int) -> dict:
    params = {}
    params["layer_type"] = "dropout"
    params["name"] = name
    params["idx"] = idx
    params["probability"] = layer.p
    return params

@dataclass
class Parser:
    path: str
    data: dict = field(default_factory=dict)
    idx: int = 0
    module_idx: int = 0

    def parse_module(self, module, name: str):

        if isinstance(module, (nn.ModuleDict, nn.ModuleList, nn.Sequential)):
            for (new_name, new_module) in module.named_children():
                self.parse_module(new_module, name=f"{name}.{new_name}")
            return
        elif list(module.children()) != []:
            module_name = module.__class__.__name__
            custom_module_name = f"{module_name}_{self.module_idx}"
            self.module_idx += 1
            for (new_name, new_module) in module.named_children():
                self.parse_module(new_module, name=f"{custom_module_name}.{new_name}")
            return
        elif isinstance(module, nn.ReLU):
            data = save_relu(module, name=name, idx=self.idx)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            data = save_adaptiveavgpool2d(module, name=name, idx=self.idx)
        elif isinstance(module, nn.Conv2d):
            data = save_conv2d(module, name=name, path=self.path, idx=self.idx)
        elif isinstance(module, nn.BatchNorm2d):
            data = save_batchnorm2d(module, name=name, path=self.path, idx=self.idx)
        elif isinstance(module, nn.Dropout):
            data = save_dropout(module, name=name, idx=self.idx)
        elif isinstance(module, nn.MaxPool2d):
            data = save_maxpool2d(module, name=name, idx=self.idx)
        else:
            raise NotImplementedError(type(module))
        self.idx += 1
        self.data["layers"].append(data)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module