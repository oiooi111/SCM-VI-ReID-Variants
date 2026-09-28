import random

import torch
import torch.nn as nn


class CrossModalityMixStyle(nn.Module):
    """Mix low-level feature statistics across RGB and IR samples."""

    def __init__(self, probability=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.probability = probability
        self.alpha = alpha
        self.eps = eps

    def forward(self, features, rgb_count):
        if not self.training or self.probability <= 0 or random.random() > self.probability:
            return features
        batch_size = features.size(0)
        ir_count = batch_size - rgb_count
        if rgb_count <= 0 or ir_count <= 0:
            return features
        original_dtype = features.dtype
        statistics = features.float()
        mean = statistics.mean(dim=(2, 3), keepdim=True)
        std = (statistics.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        normalized = (statistics - mean) / std
        rgb_reference = rgb_count + torch.arange(rgb_count, device=features.device) % ir_count
        ir_reference = torch.randperm(rgb_count, device=features.device)[:ir_count]
        reference = torch.cat([rgb_reference, ir_reference], dim=0)
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        mixing = beta.sample((batch_size, 1, 1, 1)).to(features.device)
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[reference]
        mixed_std = mixing * std + (1.0 - mixing) * std[reference]
        return (normalized * mixed_std + mixed_mean).to(original_dtype)
