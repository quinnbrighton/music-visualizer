"""ConvNeXt backbone of Vocos (Siuzdak 2023), vendored so inference does not need the vocos package or a pretrained download.
Only the non-conditional path the codec uses; parameter names match vocos.models.VocosBackbone, so its state_dict keys load unchanged.
Source: https://github.com/gemelo-ai/vocos (MIT License, Copyright (c) 2023 Charactr Inc.)"""
import torch, torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale_init_value):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim); self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim); self.act = nn.GELU(); self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):                                                   # (B,C,T) -> (B,C,T)
        h = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x).transpose(1, 2))))); return x + (self.gamma * h).transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, input_channels, dim, intermediate_dim, num_layers):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, 7, padding=3); self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim, 1 / num_layers) for _ in range(num_layers)]); self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):                                                   # (B,Cin,T) -> (B,T,dim)
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for b in self.convnext: x = b(x)
        return self.final_layer_norm(x.transpose(1, 2))
