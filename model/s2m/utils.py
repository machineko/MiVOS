# Credit: https://github.com/VainF/DeepLabV3Plus-Pytorch
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
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



class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.save = True
        
    def forward(self, x):
        if self.save:
            self.save_all()
            self.save = False
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    @torch.no_grad()
    def save_all(self):
        backbone = {"layers": []}
        idx = 0
        parser = Parser(path="backbone", data=backbone)
        for (name, module) in self.backbone.named_children():
            parser.parse_module(module, name=name)
        with open("backbone.json", "w") as f:
            ujson.dump(parser.data, f)
        print("Xd")
        return
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
