# Ball Detection (YOLO) — Pipeline Kullanımı

Bu modül, YOLO tabanlı **Player / Ball / Referee** detection modelini eğitmek ve inference almak için yapılandırma tabanlı bir pipeline sağlar.

## 1) Giriş Noktası (Entry Point)

- Ana orchestrator: `yolo/main.py`

## 2) Kurulum

PowerShell:
```powershell
cd .\model-training\ball-detection\yolo
pip install -r .\requirements.txt
```

## 3) Konfigürasyon

YAML konfigler:
- `yolo/configs/paths.yaml`
  - dataset ve output path’leri
- `yolo/configs/yolo_params.yaml`
  - eğitim hiperparametreleri + `project_name`
- `yolo/configs/extraction.yaml`
  - dataset extraction (bu projede default `run_extraction: false`)

Önemli not:
- Varsayılan yapı **hazır dataset** (`ballDataset/`) kullanacak şekilde ayarlı:
  - `paths.yaml` → `dataset_yaml: ballDataset/data.yaml`
  - `extraction.yaml` → `run_extraction: false`

## 4) Eğitim (Training)

### Tam pipeline (default)
```powershell
cd .\model-training\ball-detection\yolo
python .\main.py
```

### Sadece train (dataset hazırsa)
```powershell
python .\main.py --train-only
```

### Hızlı deneme
```powershell
python .\main.py --epochs 5 --fraction 0.01 --batch 8
```

## 5) Değerlendirme / Inference

- Evaluate:
```powershell
python .\main.py --evaluate
```

- Predict:
```powershell
python .\main.py --predict "C:\path\to\images_or_video"
```

## 6) Çıktılar (Outputs)

Varsayılan olarak:
- Model ağırlıkları: `models/player_ball_detector/weights/best.pt`
- Eğitim metrikleri: `models/player_ball_detector/results.csv`
- Raporlar: `reports/player_ball_detector/...`

## 7) Pipeline Entegrasyonu

Bu modülün ana çıktısı olan `best.pt` şu aşamalarda kullanılır:
- Tracking + ReID runner: `model-training/tracking-reid-osnet/config.yaml`
  - `detector_weights: .../ball-detection/models/player_ball_detector/weights/best.pt`

Pipeline’da önerilen minimum contract:
- Input: YOLO dataset YAML (`ballDataset/data.yaml`)
- Output: `best.pt` (detector weights)
