"""
InceptionV3 backbone wrapper for MCPNet.

Returns 4 spatial feature maps (l1, l2, l3, l4) matching MCPNet's expected interface.
Uses forward hooks to extract intermediate feature maps at 4 semantic stages.

Hook points and channel dimensions (input 299x299):
  l1: after Conv2d_4a_3x3  → 192 ch, ~71x71
  l2: after Mixed_5d       → 288 ch, 35x35
  l3: after Mixed_6e       → 768 ch, 17x17
  l4: after Mixed_7c       → 2048 ch, 8x8

With concept_cha=32: concepts per layer = [6, 9, 24, 64]
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class InceptionWrapper(nn.Module):
    def __init__(self, inception):
        super().__init__()
        self.model = inception
        self.model.aux_logits = False
        self._features = {}

        self.model.Conv2d_4a_3x3.register_forward_hook(self._hook('l1'))
        self.model.Mixed_5d.register_forward_hook(self._hook('l2'))
        self.model.Mixed_6e.register_forward_hook(self._hook('l3'))
        self.model.Mixed_7c.register_forward_hook(self._hook('l4'))

    def _hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def forward(self, x):
        self._features = {}
        _ = self.model(x)
        return (
            self._features['l1'],
            self._features['l2'],
            self._features['l3'],
            self._features['l4'],
        )


def load_model(model_name, num_classes):
    inception = tvm.inception_v3(weights=None, aux_logits=False)
    return InceptionWrapper(inception)


def load_pretrained(path, model):
    """
    Load pretrained torchvision InceptionV3 state_dict.
    Keys are remapped from torchvision format to InceptionWrapper's model.* namespace.
    fc and AuxLogits parameters are ignored (strict=False).
    """
    state = torch.load(path, map_location='cpu')
    new_state = {f'model.{k}': v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    # Only fc / AuxLogits should be missing/unexpected
    non_fc_missing = [k for k in missing if 'fc' not in k and 'AuxLogits' not in k]
    if non_fc_missing:
        print(f'[inception_net] WARNING: unexpected missing keys: {non_fc_missing[:5]}')
    return model


def load_weight(path, model):
    data = torch.load(path, map_location='cpu')
    state = data['Model']
    model.load_state_dict(state, strict=True)
    return model


if __name__ == '__main__':
    m = load_model('inceptionv3', num_classes=200).cuda()
    x = torch.randn(2, 3, 299, 299).cuda()
    l1, l2, l3, l4 = m(x)
    print(l1.shape, l2.shape, l3.shape, l4.shape)
    # Expected: [2,192,71,71] [2,288,35,35] [2,768,17,17] [2,2048,8,8]
