"""Transolver 风格的学习分块：点特征到固定数量 token 的软分配。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedTokenizer(nn.Module):
    """以可学习 token 查询聚合点云，并用求积权重计算 token 与形心。"""

    def __init__(self, input_dim, d_model, num_tokens, temperature=0.5):
        super().__init__()
        self.num_tokens = num_tokens
        self.scale = math.sqrt(d_model)
        self.assignment_key = nn.Linear(input_dim, d_model, bias=False)
        self.token_embed = nn.Parameter(torch.empty(num_tokens, d_model))
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        nn.init.trunc_normal_(self.token_embed, std=0.02)

    def forward(self, features, weights):
        """features (N,C) 前两列为坐标，weights (N,) 为点求积权重。"""
        keys = self.assignment_key(features)
        logits = keys @ self.token_embed.T
        logits = logits / (self.scale * self.temperature.clamp_min(0.05))
        assignment = F.softmax(logits, dim=-1)
        weighted_assignment = assignment * weights[:, None]
        mass = weighted_assignment.sum(dim=0)
        tokens = weighted_assignment.T @ features / mass[:, None]
        return tokens, mass, tokens[:, :2]
