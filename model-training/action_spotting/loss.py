# src/loss.py
import torch
import torch.nn as nn

class SpottingLoss(nn.Module):
    def __init__(self):
        super(SpottingLoss, self).__init__()
        # Soft etiketler (0.8, 0.5 gibi) için BCE Loss harikadır.
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_gaussian):
        """
        pred_logits: (Batch, Classes, Time) -> Modelin her kare için tahmini
        target_gaussian: (Batch, Classes, Time) -> Bizim ürettiğimiz Gaussian tepeciği
        """
        loss = self.bce(pred_logits, target_gaussian)
        return loss