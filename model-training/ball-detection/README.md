# Ball Detection Module

Bu modül, futbol maçlarında **oyuncu**, **top** ve **hakem** tespiti için YOLO11n tabanlı transfer learning kullanır.

## 📊 Dataset Bilgileri

- **Training Images:** 8,677 images
- **Test Images:** 2,996 images  
- **Classes:** 3 sınıf
  - `Player (0)`: 86% - Oyuncular
  - `Ball (1)`: 5.7% - Top
  - `Referee (2)`: 8.4% - Hakemler

## 🚀 Hızlı Başlangıç

### Gereksinimler
```bash
cd model-training/ball-detection/yolo
pip install -r requirements.txt
```

### Temel Kullanım

```bash
# Tam training (tüm dataset)
python main.py --epochs 50

# Hızlı test (az veri ile)
python main.py --epochs 5 --fraction 0.01 --batch 8

# GPU ile training
python main.py --epochs 50 --device cuda:0

# Intel iGPU ile (Linux)
python main.py --epochs 50 --device xpu

# CPU ile düşük batch size
python main.py --epochs 20 --batch 4 --device cpu
```

### Parametreler

| Parametre | Açıklama | Varsayılan |
|-----------|----------|-----------|
| `--epochs` | Training epoch sayısı | 50 |
| `--batch` | Batch size | 16 |
| `--imgsz` | Görüntü boyutu | 1024 |
| `--device` | Cihaz (`cuda:0`, `cpu`, `xpu`, `dml`) | auto |
| `--fraction` | Dataset'in kullanılacak kısmı (0.0-1.0) | 1.0 |
| `--no-extraction` | Veri çıkarımını atla | False |

## 📁 Proje Yapısı

```
ball-detection/
├── ballDataset/                    # Ana dataset (gitignore'da)
│   ├── data.yaml                  # YOLO dataset konfigürasyonu
│   ├── images/                    # Görüntüler
│   │   ├── train/                 # Training görüntüleri
│   │   └── test/                  # Test görüntüleri
│   └── labels/                    # YOLO format etiketleri
│       ├── train/                 # Training etiketleri
│       └── test/                  # Test etiketleri
├── yolo/                          # Ana kod dizini
│   ├── main.py                    # Ana çalıştırma scripti
│   ├── configs/                   # Konfigürasyon dosyaları
│   ├── modules/                   # Core modüller
│   ├── utils/                     # Yardımcı fonksiyonlar
│   └── reports/                   # Rapor oluşturma
├── models/                        # Training çıktıları (gitignore'da)
└── reports/                       # Rapor dosyaları (gitignore'da)
```

## 🎯 Model Performansı

### Transfer Learning Baseline
- **Kaynak Model:** Player Detection (optimize edilmiş)
- **Hedef Classes:** Player + Ball + Referee 
- **mAP50 Baseline:** 0.211 (sadece player detection)

### Expected Improvements
- **Player Detection:** Yüksek performans korunur (mAP50 > 0.8)
- **Ball Detection:** Zorlu sınıf (küçük objeler, class imbalance)
- **Referee Detection:** Orta zorluk (player'a benzer şekiller)

## ⚡ Performans Optimizasyonu

### RTX 5070 Settings
```bash
python main.py --epochs 50 --batch 16 --workers 12 --device cuda:0
```

### Intel iGPU (Linux)
```bash
# Intel Extension for PyTorch gerekli
pip install intel-extension-for-pytorch
python main.py --epochs 50 --batch 8 --device xpu
```

### CPU Optimization
```bash
python main.py --epochs 20 --batch 4 --workers 8 --device cpu
```

## 📈 Training Monitoring

Training sırasında şu dosyalar oluşturulur:

- `models/player_ball_detector/weights/best.pt` - En iyi model
- `models/player_ball_detector/results.csv` - Training metrikleri
- `reports/player_ball_detector/[timestamp]/` - Detaylı raporlar

## 🔧 Troubleshooting

### Common Issues

**1. GPU Memory Error**
```bash
# Batch size'ı azalt
python main.py --batch 8 --epochs 30
```

**2. Slow Training on CPU**
```bash
# Fraction kullanarak az veri ile test et
python main.py --fraction 0.01 --epochs 5 --batch 2
```

**3. Dataset Not Found**
```bash
# Dataset'in ballDataset klasöründe olduğundan emin ol
ls ballDataset/data.yaml
```

**4. Poor Ball Detection**
```bash
# Görüntü boyutunu arttır (top küçük objeler için)
python main.py --imgsz 1280 --epochs 50
```

## 📊 Dataset Visualization

```bash
# Label kalitesini kontrol et
python visualize_labels.py

# Belirli sınıfı incele
python visualize_specific_class.py --class_id 1  # Ball class
```

## 🎛️ Configuration Files

### `configs/yolo_params.yaml`
YOLO training hiperparametreleri

### `configs/device.yaml` 
Cihaz öncelikleri ve ayarları

### `configs/paths.yaml`
Dataset ve model yolları

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLOv8/v11
- OpenCV
- NumPy
- Pandas
- Matplotlib

## 🤝 Contributing

1. Yeni özellikler için branch oluştur
2. Dataset güncellemeleri için `ballDataset/data.yaml` kontrol et
3. Model performansı için `reports/` klasörünü incele

## 📝 License

Bu proje FoMAC (Football Match Analysis and Classification) projesi kapsamındadır.