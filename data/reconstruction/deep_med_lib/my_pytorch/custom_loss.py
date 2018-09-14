import torch
import torch.nn as nn

def get_objective_loss(objective_loss):
    if objective_loss == 'weighted_cross_entropy':
        weight = torch.Tensor([.1, 1., 1., 1.])
        criterion = CrossEntropyLoss2d(weight)
    elif objective_loss == 'bce':  # binary cross entropy
        criterion = nn.BCELoss()
    else: # default: L2
        criterion = nn.MSELoss()
    return criterion


# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


