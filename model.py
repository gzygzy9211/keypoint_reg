import torch
from torchvision.models import resnet
from torch import nn, Tensor
from typing import Callable, Dict, Any
from functools import partial


class ResNetBackbone(resnet.ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        print(block, layers)
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)
        self.remove_origin_head()

    def remove_origin_head(self):
        self.fc = None
        self.avgpool = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet12(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert not pretrained
    model = resnet.ResNet(resnet.BasicBlock, [1, 1, 2, 1], **kwargs)
    return model


def resnet_process(getter: Callable[..., resnet.ResNet], **kwargs):
    resnet = getter(**kwargs)
    backbone = ResNetBackbone.__new__(ResNetBackbone)
    backbone.__dict__ = resnet.__dict__.copy()
    backbone.remove_origin_head()
    del resnet
    return backbone


BACKBONS = {
    'resnet12': partial(resnet_process, resnet12),
    'resnet18': partial(resnet_process, resnet.resnet18),
    'resnet34': partial(resnet_process, resnet.resnet34),
    'resnet50': partial(resnet_process, resnet.resnet50),
}


def get_activation(channels: int, t: str = 'relu', kwargs: Dict[str, Any] = {}) -> nn.Module:
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'prelu': nn.PReLU,
        'leakyrelu': nn.LeakyReLU,
    }
    assert t in ACTIVATIONS
    return ACTIVATIONS['prelu'](channels, **kwargs) if t == 'prelu' \
        else ACTIVATIONS[t](**kwargs)


class LinearBNAct(nn.Module):

    def __init__(self, in_dim: int, out_dim: int,
                 activation: str = 'relu', activation_kwargs: Dict[str, Any] = {}) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = get_activation(out_dim, activation, activation_kwargs)
        assert isinstance(self.act, nn.Module)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = x.unsqueeze(dim=2)
        x = self.bn(x)
        x = self.act(x)
        return x.squeeze(dim=2)


class KeyPointModel(nn.Module):

    def __init__(self, num_pt: int, input_size: int = 224,
                 hidden_dim: int = 256, num_hidden: int = 1,
                 backbone: str = 'resnet18', backbone_kwargs: Dict[str, Any] = {},
                 hidden_act_type: str = 'relu', hidden_act_kwargs: Dict[str, Any] = {}):
        super().__init__()
        assert backbone in BACKBONS
        self.backbone = BACKBONS[backbone](**backbone_kwargs)
        assert isinstance(self.backbone, nn.Module)

        assert input_size > 0 and input_size % 32 == 0
        self.input_size = input_size
        assert hidden_dim > 0
        self.hidden_dim = hidden_dim
        assert num_hidden > 0
        self.num_hidden = num_hidden
        assert num_pt > 0
        self.num_pt = num_pt

        try:
            device_bak = next(self.backbone.parameters()).device
            training = self.backbone.training
            self.backbone.cpu()
            self.backbone.eval()
            out = self.backbone(torch.rand(1, 3, input_size, input_size, dtype=torch.float32))
            out_size = out.numel()
        finally:
            self.backbone.to(device_bak).train(training)

        # assume backbone is stride 32
        self.hiddens = nn.Sequential(
            nn.Flatten(1),
            LinearBNAct(out_size, hidden_dim,
                        hidden_act_type, hidden_act_kwargs),
            *[LinearBNAct(hidden_dim, hidden_dim, hidden_act_type,
                          hidden_act_kwargs) for _ in range(num_hidden - 1)],
            nn.Linear(hidden_dim, 2 * num_pt, bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.hiddens(x)
        return x


def build_model() -> KeyPointModel:
    from config import cfg

    return KeyPointModel(
        cfg.DEFINITION.NUM_PT,
        cfg.MODEL.INPUT_SIZE,
        cfg.MODEL.HIDDEN_DIM,
        cfg.MODEL.NUM_HIDDEN,
        cfg.MODEL.BACKBONE, {'pretrained': True},
        cfg.MODEL.HIDDEN_ACT,
        cfg.MODEL.HIDDEN_ACT_KWARGS,
    )


if __name__ == '__main__':

    import sys
    from config import merge_from_file

    merge_from_file(sys.argv[1])
    model = build_model()
    model.eval()

    x = torch.rand((1, 3, model.input_size, model.input_size), dtype=torch.float32)

    print(model)
    print(model(x))
