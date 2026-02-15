# Model Training

Bu klasör FoMAC projesinin makine öğrenmesi modellerini içerir. Futbol maçı analizi için nesne tespiti modellerini eğitir.

## 📁 Modül Yapısı

### 🎯 [Player Detection](./player-detection/)
Futbol oyuncularını tespit eden temel model
- **Model:** YOLO11n 
- **Classes:** Player detection (optimized)
- **Status:** ✅ Production ready
- **Performance:** High accuracy player detection

### ⚽ [Ball Detection](./ball-detection/)  
Oyuncu, top ve hakem tespiti yapan çok sınıflı model
- **Model:** YOLO11n (transfer learning from player detection)
- **Classes:** Player, Ball, Referee
- **Status:** 🔄 In development
- **Performance:** Challenging ball detection due to small size

### 🛰️ [Tracking & ReID](./tracking/)
SoccerNet ReID + SNMOT tracking entegrasyonu
- **Detector:** `models/player_ball_detector/weights/best.pt`
- **ReID:** ResNet50 (SoccerNet fine-tune)
- **Tracker:** StrongSORT/ByteTrack (Ultralytics)
- **Çıktı:** MOTChallenge formatında track dosyaları

## 🚀 Hızlı Başlangıç

```bash
# Player detection training
cd player-detection
python main.py --epochs 50

# Ball detection training (multi-class)
cd ball-detection/yolo  
python main.py --epochs 50
```

## 🎛️ Sistem Gereksinimleri

### Minimum
- **CPU:** 4+ cores
- **RAM:** 8GB+
- **Storage:** 10GB+ free space
- **Python:** 3.8+

### Recommended
- **GPU:** RTX 3060+ or RTX 5070
- **RAM:** 16GB+
- **Storage:** SSD with 50GB+ free
- **CPU:** Intel i7+ or AMD Ryzen 7+

### Intel iGPU Support (Linux)
- **Iris Xe Graphics** or newer
- Intel Extension for PyTorch
- Level Zero drivers

## ⚡ Performans Optimizasyonu

### GPU Training
```bash
# NVIDIA GPU
python main.py --device cuda:0 --batch 16

# Intel iGPU (Linux)
python main.py --device xpu --batch 8
```

### CPU Training
```bash
# Multi-core CPU optimization
python main.py --device cpu --batch 4 --workers 8
```

## 📊 Dataset Bilgileri

| Dataset | Images | Classes | Status |
|---------|---------|---------|---------|
| Player Detection | ~10K | 1 (Player) | ✅ Ready |
| Ball Detection | 11,673 | 3 (Player, Ball, Referee) | ✅ Ready |

## 🔧 Configuration

Her modül kendi konfigürasyonlarına sahiptir:
- `configs/device.yaml` - Hardware settings
- `configs/yolo_params.yaml` - Training parameters  
- `configs/paths.yaml` - Dataset paths

## 📈 Training Monitoring

Training sırasında otomatik olarak şunlar oluşturulur:
- **Weights:** `models/*/weights/best.pt`
- **Metrics:** `models/*/results.csv`  
- **Reports:** Excel + HTML raporlar
- **Logs:** Detaylı training logları

## 🎯 Model Performance

### Player Detection
- **mAP50:** 0.85+ (production)
- **Inference Speed:** ~10ms (RTX 5070)
- **Use Case:** Base model for transfer learning

### Ball Detection (Multi-class)
- **Player mAP50:** 0.80+ (transfer learning)
- **Ball mAP50:** 0.30+ (challenging small objects)
- **Referee mAP50:** 0.60+ (medium difficulty)

## 🛠️ Development Workflow

1. **Dataset Preparation**
   ```bash
   # Check dataset structure
   python visualize_labels.py
   ```

2. **Model Training**
   ```bash
   # Quick test
   python main.py --fraction 0.01 --epochs 5
   
   # Full training
   python main.py --epochs 50
   ```

3. **Performance Analysis**
   ```bash
   # Check reports folder
   ls reports/*/
   ```

## 📋 Dependencies

```bash
# Core ML libraries
pip install ultralytics torch torchvision

# Data processing
pip install opencv-python pandas numpy matplotlib

# Optional: Intel GPU support
pip install intel-extension-for-pytorch

# Optional: Advanced visualization  
pip install seaborn plotly
```

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
# Reduce batch size
python main.py --batch 8 --epochs 30
```

**Slow CPU Training**
```bash
# Use data fraction for testing
python main.py --fraction 0.001 --epochs 2 --batch 2
```

**Intel iGPU Not Detected**
```bash
# Install Intel Extension
pip install intel-extension-for-pytorch
# Check GPU availability
python -c "import intel_extension_for_pytorch as ipex; print(ipex.xpu.is_available())"
```

**Poor Ball Detection**
```bash
# Increase image size for small objects
python main.py --imgsz 1280 --epochs 50
```

## 🔄 Transfer Learning Pipeline

```
Player Detection (Base) 
        ↓
Ball Detection (Multi-class)
        ↓  
Advanced Analysis Models
```

## 📁 Directory Structure

```
model-training/
├── player-detection/          # Base player detection
│   ├── main.py
│   ├── configs/
│   ├── modules/
│   └── models/               # Output models
├── ball-detection/           # Multi-class detection
│   ├── ballDataset/         # Dataset (gitignored)
│   ├── yolo/               # Source code
│   └── models/             # Output models  
├── tracking/               # ReID + MOT pipeline
│   ├── configs/
│   ├── modules/
│   ├── run_tracking.py
│   └── train_reid.py
└── yolo/                    # Shared YOLO utilities
```

## 🤝 Contributing

1. Her modül için ayrı branch kullan
2. Dataset değişiklikleri için `data.yaml` güncelle
3. Performance regressions için rapor kontrol et
4. Model weights'i commit etme (gitignore'da)

## 📄 Documentation

- [Player Detection README](./player-detection/README.md)
- [Ball Detection README](./ball-detection/README.md)
- [Tracking & ReID README](./tracking/README.md)
- [YOLO Configuration Guide](./ball-detection/yolo/README.md)