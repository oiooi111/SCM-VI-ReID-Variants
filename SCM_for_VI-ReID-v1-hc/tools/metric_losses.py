import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroCenterTripletLoss(nn.Module):
    """Batch-hard triplet loss over RGB/IR identity centers."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, rgb_features, rgb_labels, ir_features, ir_labels):
        centers = []
        center_labels = []
        modalities = []

        for identity in torch.unique(rgb_labels):
            rgb_mask = rgb_labels == identity
            ir_mask = ir_labels == identity
            if not torch.any(ir_mask):
                continue
            centers.append(rgb_features[rgb_mask].mean(dim=0))
            centers.append(ir_features[ir_mask].mean(dim=0))
            center_labels.extend([identity, identity])
            modalities.extend([0, 1])

        if len(centers) < 4:
            return rgb_features.sum() * 0.0

        centers = F.normalize(torch.stack(centers).float(), p=2, dim=1)
        center_labels = torch.stack(center_labels)
        modalities = torch.tensor(modalities, device=centers.device)
        distance = 1.0 - centers @ centers.t()

        same_id = center_labels[:, None].eq(center_labels[None, :])
        cross_modal = modalities[:, None].ne(modalities[None, :])
        positive_mask = same_id & cross_modal
        negative_mask = ~same_id

        positive = distance.masked_fill(~positive_mask, float('-inf')).max(dim=1).values
        negative = distance.masked_fill(~negative_mask, float('inf')).min(dim=1).values
        valid = torch.isfinite(positive) & torch.isfinite(negative)
        if not torch.any(valid):
            return centers.sum() * 0.0
        return F.relu(positive[valid] - negative[valid] + self.margin).mean()
