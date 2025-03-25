import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DisNet(nn.Module):
    def __init__(self, dim, att_dim, num_heads=2, qkv_bias=True):
        super().__init__()
        self.dim = dim
        encoder = []
        encoder_dim = att_dim
        for _ in range(2):
            encoder.append(nn.Linear(encoder_dim, dim * 2, bias=qkv_bias))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = dim * 2
        encoder.append(nn.Linear(encoder_dim, dim, bias=qkv_bias))
        self.encoder = nn.Sequential(*encoder)

        self.att_dim = att_dim
        self.num_heads = num_heads
        self.head_dim = dim
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=qkv_bias)


        decoder = []
        decoder_dim = dim
        for _ in range(2):
            decoder.append(nn.Linear(decoder_dim, dim * 2, bias=qkv_bias))
            decoder.append(nn.LeakyReLU(0.2, inplace=True))
            decoder_dim = dim * 2
        decoder.append(nn.Linear(decoder_dim, att_dim, bias=qkv_bias))
        self.decoder = nn.Sequential(*decoder)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, inputs):
        hidden_feat = self.encoder(inputs)
        B, N, C = hidden_feat.shape
        qkv = self.qkv(hidden_feat).reshape(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_1, attn_2 = attn.unbind(1)
        attn_1 = attn_1.softmax(dim=-1)
        attn_2 = attn_2.softmax(dim=-1)

        v1, v2 = v.unbind(1)


        z_1 = attn_1 @ v1
        z_2 = attn_2 @ v2

        output_1 = self.decoder(z_1)
        output_2 = self.decoder(z_2)

        if self.training:
            recon_loss = F.mse_loss(inputs, output_1) + F.mse_loss(inputs, output_2)
            dis_loss = torch.mean(self.cos(attn_1.reshape((B, N ** 2)), attn_2.reshape((B, N ** 2))))
            return recon_loss, dis_loss
        else:
            anomaly_score = (F.mse_loss(inputs, output_1, reduction='none') + F.mse_loss(inputs, output_2, reduction='none')).sum(dim=[1,2])
            return anomaly_score