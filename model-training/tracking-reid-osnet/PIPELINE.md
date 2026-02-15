# Tracking + Team ReID (tracking-reid-osnet) — Pipeline Kullanımı

Bu modül, YOLO detections + ReID embedding + takım rengi füzyonu ile **BoT-SORT benzeri tracking** üretir.

## 1) Önerilen Giriş Noktası

- Modern runner: `run_botsort_team_reid.py`

(Eski/alternatif demo script)
- `run_tracker_with_teams_with_reid.py` (boxmot/ByteTrack tabanlı, daha “hardcoded”)

## 2) Kurulum

PowerShell:
```powershell
cd .\model-training\tracking-reid-osnet
pip install -r .\requirements.txt
```

Notlar:
- `requirements.txt` içinde `ultralytics`, `opencv-python`, `pyyaml` var.
- `osnet.enabled: true` ise ek olarak OSNet weights gerekir (aşağıda).

## 3) Konfigürasyon (config.yaml)

Varsayılan config: `config.yaml`

Kritik alanlar:
- `video`: input maç videosu (ör: `.../1_720p.mkv`)
- `detector_weights`: YOLO weights (`best.pt`)
  - Bu genelde `model-training/ball-detection/models/player_ball_detector/weights/best.pt`
- `reid_weights`: ReID model weights (repo dışında olabilir)
- `device`: `cuda:0` / `cpu`
- `save_video`: opsiyonel çıktı video path
- `save_txt`: CSV çıktı path

OSNet stage:
- `osnet.enabled: true`
- `osnet.weights: "weights/osnet_x1_0_imagenet.pth"` (config’e göre relative)

## 4) Çalıştırma

### Sıfır argüman (config.yaml ile)
```powershell
cd .\model-training\tracking-reid-osnet
python .\run_botsort_team_reid.py
```

### Config override ile
```powershell
python .\run_botsort_team_reid.py `
  --config .\config.yaml `
  --video "C:\\path\\to\\match.mp4" `
  --device cuda:0 `
  --save_video .\outputs\\tracked.mp4 `
  --save_txt .\outputs\\tracks.csv
```

## 5) Çıktılar (Outputs)

- CSV (`save_txt`):
  - Kolonlar: `frame_id,track_id,cls_id,conf,x1,y1,x2,y2,team_id,...`
- Video (`save_video`): çizimli tracking çıktısı

## 6) Pipeline Entegrasyonu

Minimum pipeline contract:
- Input: maç videosu + YOLO detector weights + ReID weights
- Output: per-frame track CSV (ve opsiyonel video)

Bu modülün YOLO detector bağımlılığı genelde ball-detection çıktısından gelir:
- `model-training/ball-detection/models/player_ball_detector/weights/best.pt`

Not: `config.yaml` içindeki default path’ler makineye özgü absolute path olabilir; pipeline’da stabil olması için bu path’leri **relative** tutup `--config` ile yönetmek en kolay yöntem.
