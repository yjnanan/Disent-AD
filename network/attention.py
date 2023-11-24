import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, dim, att_dim, num_heads=2, qkv_bias=True, attn_drop=0.0):
        super().__init__()
        self.map = nn.Linear(att_dim, dim)
        self.att_dim = att_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_1 = nn.Dropout(attn_drop)
        self.attn_drop_2 = nn.Dropout(attn_drop)

        self.recon_1 = nn.Linear(dim // 2, att_dim)
        self.recon_2 = nn.Linear(dim // 2, att_dim)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, inputs, labels=None):
        hidden_feat = self.map(inputs)
        B, N, C = hidden_feat.shape
        qkv = self.qkv(hidden_feat).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_1, attn_2 = attn.unbind(1)
        attn_1 = attn_1.softmax(dim=-1)
        attn_2 = attn_2.softmax(dim=-1)
        attn_1 = self.attn_drop_1(attn_1)
        attn_2 = self.attn_drop_2(attn_2)

        v1, v2 = v.unbind(1)

        recon_feat_1 = attn_1 @ v1
        recon_feat_2 = attn_2 @ v2

        output_1 = self.recon_1(recon_feat_1)
        output_2 = self.recon_2(recon_feat_2)

        if self.training:
            recon_loss = F.mse_loss(output_1, inputs) + F.mse_loss(output_2, inputs)
            cos_loss = torch.mean(-self.cos(v1.reshape((B, self.head_dim * N)), v2.reshape((B, self.head_dim * N))))
            dis_loss = torch.mean(self.cos(attn_1.reshape((B, N ** 2)), attn_2.reshape((B, N ** 2))))

            return recon_loss, cos_loss, dis_loss
        else:
            anomaly_score = self.cos(v1.reshape((B, self.head_dim * N)), v2.reshape((B, self.head_dim * N)))
            return anomaly_score