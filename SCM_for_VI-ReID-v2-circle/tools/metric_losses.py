import torch
import torch.nn as nn
import torch.nn.functional as F


class CirclePairLoss(nn.Module):
    """Pair-based Circle loss for identity-labelled ReID embeddings."""

    def __init__(self, margin=0.25, gamma=128.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, features, labels):
        features = F.normalize(features.float(), p=2, dim=1)
        similarity = features @ features.t()
        same_id = labels[:, None].eq(labels[None, :])
        same_id.fill_diagonal_(False)
        different_id = ~labels[:, None].eq(labels[None, :])

        losses = []
        delta_p = 1.0 - self.margin
        delta_n = self.margin
        for row in range(features.size(0)):
            positive = similarity[row][same_id[row]]
            negative = similarity[row][different_id[row]]
            if positive.numel() == 0 or negative.numel() == 0:
                continue
            alpha_p = F.relu(1.0 + self.margin - positive.detach())
            alpha_n = F.relu(negative.detach() + self.margin)
            logit_p = -self.gamma * alpha_p * (positive - delta_p)
            logit_n = self.gamma * alpha_n * (negative - delta_n)
            losses.append(F.softplus(torch.logsumexp(logit_p, dim=0) +
                                     torch.logsumexp(logit_n, dim=0)))

        if not losses:
            return features.sum() * 0.0
        return torch.stack(losses).mean()
