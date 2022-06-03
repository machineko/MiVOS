from dataclasses import dataclass, field

import torch
import torch.nn as nn
try:
    from torchvision.models.utils import load_state_dict_from_url
except ModuleNotFoundError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['ResNet', 'resnet50']

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
    np.save(save_path + "_running_var.npy", layer.running_var.numpy())
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

    def parse_module(self, module, name: str):

        if isinstance(module, (nn.ModuleDict, nn.ModuleList, nn.Sequential)):
            for (new_name, new_module) in module.named_children():
                self.parse_module(new_module, name=f"{name}.{new_name}")
            return
        elif list(module.children()) != []:
            module_name = module.__class__.__name__
            print(module_name, "Not supported")
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


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
