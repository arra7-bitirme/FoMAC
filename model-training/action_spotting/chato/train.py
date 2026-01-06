import torch
from torch.utils.data import DataLoader
from dataset import SoccerNetDataset
from model import ActionSpottingTransformerNet
from loss import ActionSpottingLoss

# =====================
# AYARLAR
# =====================
DATASET_PATH = "C:/FoMAC_Dataset/action_spotting"
BATCH_SIZE = 8          # Windows + RAM dostu
EPOCHS = 10             # İlk test için düşük
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# DATASET
# =====================
dataset = SoccerNetDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# =====================
# MODEL
# =====================
model = ActionSpottingTransformerNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = ActionSpottingLoss()

# =====================
# TRAIN LOOP
# =====================
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for x, y_cls in loader:
        x = x.to(device)           # (B, 40, 8576)
        y_cls = y_cls.to(device)   # (B, 40)

        optimizer.zero_grad()

        cls_pred, _ = model(x)

        # sadece orta frame'i supervise et
        center = WINDOW_SIZE // 2
        loss = criterion(cls_pred, y_cls[:, center])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")


# =====================
# MODEL KAYDET
# =====================
torch.save(model.state_dict(), "action_spotting_transformer.pth")
print("✅ Eğitim tamamlandı, model kaydedildi.")
