# Action Spotting (spotting_v2) — Pipeline Kullanımı

Bu modül SoccerNet v2 style **pre-extracted feature** (.npy) dosyaları üzerinden **olay tespiti (action spotting)** modeli eğitir ve inference yapar.

## 1) Giriş Noktaları (Entry Points)

- Eğitim: `train.py`
- Feature dosyası üzerinde inference (JSON/print): `inference.py`
- Video üstüne yazıp klip üretme: `video_inference.py`

## 2) Veri / Input Beklentisi

### Dataset yapısı
Kod, `config.py` içindeki `DATASET_DIR` altında tüm klasörleri tarar ve her maç klasöründe şunları arar:
- `Labels-v2.json`
- Feature dosyaları (1. ve 2. devre için)
  - `1{FEATURE_SUFFIX}` ve `2{FEATURE_SUFFIX}`

Varsayılanlar (config):
- `DATASET_DIR = H:/soccerNet`
- `FEATURE_TYPE = "resnet"` → `FEATURE_SUFFIX = "_ResNET_TF2_PCA512.npy"` ve `FPS = 2`

Feature dosyasının shape’i `(T, FEATURE_DIM)` olmalı.

## 3) Kurulum

Bu klasör bağımsız bir `requirements.txt` içermiyor; tipik olarak şunlar gerekir:
- `torch`, `numpy`, `tqdm`
- Video inference için: `opencv-python`

## 4) Eğitim (Training)

1) [config.py] içindeki `DATASET_DIR`’i kendi SoccerNet feature root’una ayarla.
2) Gerekirse:
   - `FEATURE_TYPE` (baidu/resnet)
   - `WINDOW_SIZE_SEC`, `EPOCHS`, `BATCH_SIZE`
3) Çalıştır:

PowerShell:
```powershell
cd .\model-training\action_spotting\spotting_v2
python .\train.py
```

### Çıktılar
- Checkpoint’ler: `checkpoints/` (örn: `v3_cnn_...._best_map.pth`)
- Loglar: `logs/`

## 5) Inference (Feature .npy üstünde)

`inference.py` CLI bekliyor:

```powershell
cd .\model-training\action_spotting\spotting_v2
python .\inference.py --features "H:\soccerNet\...\1_ResNET_TF2_PCA512.npy" --checkpoint ".\checkpoints\<MODEL>.pth"
```

Notlar:
- Inference sliding-window ile çalışır.
- Threshold/NMS davranışı kod içinde (`predict()` ve `nms()`), gerekirse oradan ayarlanır.

## 6) Video Üzerine Tahmin Yazıp Klip Üretme

`video_inference.py` şu an “hardcoded path” ile çalışıyor.

1) Dosyanın en üstündeki değişkenleri güncelle:
- `VIDEO_PATH`
- `FEATURE_PATH`
- `CHECKPOINT_PATH`
- `OUTPUT_VIDEO_PATH`

2) Çalıştır:
```powershell
cd .\model-training\action_spotting\spotting_v2
python .\video_inference.py
```

Çıktı:
- `OUTPUT_VIDEO_PATH` altında event çevresi clip’ler birleştirilmiş tek video.

## 7) Pipeline Entegrasyonu (Önerilen Akış)

- (Upstream) Video → Feature extraction (bu repo içinde değil / farklı modül olabilir)
- `spotting_v2`:
  - Train: SoccerNet feature + Labels-v2.json
  - Inference: feature `.npy` → event listesi (time, label, score)

Eğer pipeline’da “tek yerden config” istiyorsan, burada kritik olanlar:
- `DATASET_DIR` (train)
- `FEATURE_TYPE` / `FEATURE_SUFFIX`
- Inference’ta kullanılan checkpoint yolu
