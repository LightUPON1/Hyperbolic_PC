# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        uncertainty = torch.ones(num, requires_grad=True)
        self.uncertainty = torch.nn.Parameter(uncertainty)

    # def forward(self, *x):
    def forward(self, x):
        loss_sum = 0
        # for i, loss in enumerate(x):
        for i, loss_name in enumerate(x):
            loss = x[loss_name]
            # loss_sum += 0.5 / (self.uncertainty[i] ** 2) * loss + torch.log(1 + self.uncertainty[i] ** 2)
            loss_sum += torch.exp(-self.uncertainty[i]) * loss + 0.5 * self.uncertainty[i]
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())