"""
ConvNeXt backbone wrapper for MCPNet.

Returns 4 spatial feature maps (l1, l2, l3, l4) matching MCPNet's expected interface.
Uses timm's features_only mode to extract per-stage outputs.

Channel dimensions per variant:
  convnext_tiny:  96, 192, 384, 768
  convnext_small: 96, 192, 384, 768
  convnext_base:  128, 256, 512, 1024
"""

import torch
import torch.nn as nn
import timm


class ConvNeXtWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # timm features_only model

    def forward(self, x):
        feats = self.backbone(x)  # list of 4 feature maps
        return feats[0], feats[1], feats[2], feats[3]


_model_pool = {
    'convnext_tiny': 'convnext_tiny',
    'convnext_small': 'convnext_small',
    'convnext_base': 'convnext_base',
}


def load_model(model_name, num_classes):
    timm_name = _model_pool[model_name.lower()]
    backbone = timm.create_model(
        timm_name,
        pretrained=False,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    return ConvNeXtWrapper(backbone)


def load_pretrained(path, model):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    return model


def load_weight(path, model):
    data = torch.load(path, map_location='cpu')
    model.load_state_dict(data['Model'], strict=True)
    return model


if __name__ == '__main__':
    m = load_model('convnext_small', num_classes=200).cuda()
    x = torch.randn(2, 3, 224, 224).cuda()
    l1, l2, l3, l4 = m(x)
    print(l1.shape, l2.shape, l3.shape, l4.shape)
    # Expected: [2,96,56,56] [2,192,28,28] [2,384,14,14] [2,768,7,7]
