import torch
import numpy as np
from model import ActionSpottingNet
from utils import nms

FPS = 2

def infer(video_features, model):
    model.eval()
    spots = []

    with torch.no_grad():
        for t in range(0, len(video_features) - 40, 10):
            x = torch.tensor(video_features[t:t+40]).unsqueeze(0).cuda()
            cls, reg = model(x)

            score, label = torch.max(torch.softmax(cls, -1), -1)
            if label.item() != 0:
                time = (t + reg.item() * 40) / FPS
                spots.append((time, score.item(), label.item()))

    return nms(spots)
