"""Predeclared loss comparisons. Every epoch is still selected by plain BCE."""
import torch
from torch import nn

from yenibot.losses import FocalLossWithLogits, RankICLoss


class AdvisorObjective(nn.Module):
    def __init__(self, spec: dict, training_targets: torch.Tensor):
        super().__init__()
        self.kind = spec['kind']
        if self.kind not in {'bce','weighted_bce','focal','bce_pearson','focal_pearson'}:
            raise ValueError('Unknown advisor loss')
        if not torch.isfinite(training_targets).all() or not ((training_targets==0)|(training_targets==1)).all():
            raise ValueError('Binary finite training targets required')
        positive = int(training_targets.sum())
        negative = len(training_targets)-positive
        if self.kind == 'weighted_bce' and min(positive,negative)==0:
            raise ValueError('Weighted BCE requires both classes in training sequences')
        weight = negative/positive if self.kind=='weighted_bce' else 1.
        self.register_buffer('pos_weight',torch.tensor(weight,dtype=torch.float32))
        self.alpha,self.gamma = float(spec.get('alpha',.6)),float(spec.get('gamma',2.))
        self.pearson_weight = float(spec.get('pearson_weight',0))
        if not 0<=self.alpha<=1 or self.gamma<0 or self.pearson_weight<0:
            raise ValueError('Invalid focal/Pearson parameters')
        self.focal = FocalLossWithLogits(alpha=self.alpha,gamma=self.gamma)
        self.pearson = RankICLoss()  # Negative Pearson; not differentiable Spearman.
        self.audit = {'kind':self.kind,'training_sequence_positives':positive,
                      'training_sequence_negatives':negative,
                      'pos_weight':weight if self.kind=='weighted_bce' else None,
                      'alpha':self.alpha if 'focal' in self.kind else None,
                      'gamma':self.gamma if 'focal' in self.kind else None,
                      'pearson_weight':self.pearson_weight,
                      'class_weight_source':'training_sequence_targets_only'}

    def forward(self, logits, labels, forward_returns):
        if self.kind=='weighted_bce':
            result = nn.functional.binary_cross_entropy_with_logits(logits,labels,pos_weight=self.pos_weight)
        elif 'focal' in self.kind:
            result = self.focal(logits,labels)
        else:
            result = nn.functional.binary_cross_entropy_with_logits(logits,labels)
        if 'pearson' in self.kind:
            result = result+self.pearson_weight*self.pearson(torch.sigmoid(logits),forward_returns)
        return result
