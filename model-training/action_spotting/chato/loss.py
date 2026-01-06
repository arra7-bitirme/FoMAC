import torch.nn as nn

class ActionSpottingLoss(nn.Module):
    def __init__(self, lambda_reg=10.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()
        self.lambda_reg = lambda_reg

    def forward(self, cls_pred, reg_pred, cls_gt, reg_gt):
        l_cls = self.cls_loss(cls_pred, cls_gt)
        l_reg = self.reg_loss(reg_pred, reg_gt)
        return l_cls + self.lambda_reg * l_reg
