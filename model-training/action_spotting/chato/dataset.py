import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

# ======================
# SABİTLER
# ======================
FPS = 2
WINDOW_SEC = 20
WINDOW_SIZE = WINDOW_SEC * FPS  # 40 frame
STRIDE = WINDOW_SIZE // 2       # %50 overlap

# SoccerNet label map
ACTION_TO_ID = {
    "Kick-off": 0,
    "Goal": 1,
    "Substitution": 2,
    "Offside": 3,
    "Shots on target": 4,
    "Shots off target": 5,
    "Clearance": 6,
    "Ball out of play": 7,
    "Foul": 8,
    "Indirect free-kick": 9,
    "Direct free-kick": 10,
    "Corner": 11,
    "Yellow card": 12,
    "Red card": 13,
    "Penalty": 14,
    "Throw-in": 15,
    "Goal kick": 16,
}

NUM_CLASSES = 17


class SoccerNetDataset(Dataset):
    def __init__(self, root_dir, feature_name="1_baidu_soccer_embeddings.npy"):
        self.windows = []

        print("Dataset taranıyor...")

        for root, _, files in os.walk(root_dir):
            if "Labels-v2.json" in files and feature_name in files:
                feat_path = os.path.join(root, feature_name)
                label_path = os.path.join(root, "Labels-v2.json")

                features = np.load(feat_path)  # (T, 8576)
                T = features.shape[0]

                cls_targets = np.zeros(T, dtype=np.int64)

                with open(label_path, "r") as f:
                    annotations = json.load(f)["annotations"]

                for event in annotations:
                    frame = int(float(event["position"]) * FPS)
                    if frame < T and event["label"] in ACTION_TO_ID:
                        cls_targets[frame] = ACTION_TO_ID[event["label"]]

                # WINDOW OLUŞTUR
                for start in range(0, T - WINDOW_SIZE, STRIDE):
                    end = start + WINDOW_SIZE
                    x = features[start:end]
                    y = cls_targets[start:end]

                    self.windows.append((
                        torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.long)
                    ))

        print(f"Toplam window sayısı: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
