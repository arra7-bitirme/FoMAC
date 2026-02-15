# Player Detection (YOLO) — Pipeline Kullanımı

Bu modül SoccerNet videolarından dataset extraction + YOLO eğitimi ile **player detection** modelini üretir.

## 1) Giriş Noktası (Entry Point)

- Ana orchestrator: `main.py`

## 2) Kurulum

PowerShell:
```powershell
cd .\model-training\player-detection
pip install -r .\requirements.txt
```

## 3) Konfigürasyon

YAML konfigler:
- `configs/paths.yaml` (kritik)
  - `workspace_root`, `soccernet_root`, `output_root`, `models_root`, `reports_root`
- `configs/extraction.yaml`
  - `run_extraction: true` (varsayılan)
  - season split’ler
- `configs/yolo_params.yaml`
  - model, epochs, imgsz, batch vs.

Windows’ta öneri:
- `configs/paths.yaml` içindeki `workspace_root: "~/..."` değerini **absolute path** yap.
- `soccernet_root` SoccerNet dataset root’unu göstermeli.

## 4) Çalıştırma

### Tam pipeline (extract + train)
```powershell
cd .\model-training\player-detection
python .\main.py
```

### Sadece extraction
```powershell
python .\main.py --extract-only
```

### Sadece training (dataset hazırsa)
```powershell
python .\main.py --train-only
```

### Hızlı deneme
```powershell
python .\main.py --epochs 5 --fraction 0.01 --batch 8
```

## 5) Çıktılar (Outputs)

- Model ağırlıkları: `models/<project_name>/weights/best.pt`
  - `project_name` varsayılanı config’te `football_detector_optimized`
- Raporlar: `reports/<project_name>/...`

## 6) Pipeline Entegrasyonu

Bu modülün `best.pt` çıktısı ball-detection fine-tune için “base model” olarak kullanılıyor:
- `model-training/ball-detection/yolo/configs/yolo_params.yaml`
  - `model: .../player-detection/models/football_detector_optimized/weights/best.pt`

Pipeline contract:
- Output: player detector weights (`best.pt`)
- Downstream: ball-detection training başlangıç ağırlığı
