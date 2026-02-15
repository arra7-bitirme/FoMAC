# FoMAC - Football Match Analysis and Classification

Futbol maçı analizi ve sınıflandırma için geliştirilmiş yapay zeka tabanlı sistem.

## 🎯 Proje Amacı

FoMAC, futbol maçlarında oyuncu, top ve hakem tespiti yaparak maç analizini otomatikleştiren bir AI sistemidir. YOLO tabanlı nesne tespiti ve transfer learning teknikleri kullanır.

## 🏗️ Sistem Mimarisi

```
FoMAC/
├── backend/                 # API ve backend servisleri
├── frontend/               # Web arayüzü
└── model-training/         # Makine öğrenmesi modelleri
    ├── player-detection/   # Oyuncu tespiti (temel model)
    └── ball-detection/     # Çok sınıflı tespit (Oyuncu+Top+Hakem)
```

## 🤖 Model Yetenekleri

### 👤 Player Detection
- **Teknoloji:** YOLO11n
- **Doğruluk:** mAP50 > 0.85
- **Durum:** ✅ Production ready
- **Kullanım:** Transfer learning için base model

### ⚽ Ball Detection (Multi-class)
- **Sınıflar:** Player (86%), Ball (5.7%), Referee (8.4%)
- **Teknoloji:** Transfer learning from player detection
- **Zorluk:** Top tespiti (küçük objeler)
- **Durum:** 🔄 Geliştirme aşamasında

## 🚀 Hızlı Başlangıç

### Model Training

```bash
# Player detection training
cd model-training/player-detection
python main.py --epochs 50

# Ball detection training
cd model-training/ball-detection/yolo
python main.py --epochs 50 --batch 16
```

### GPU Optimization

```bash
# RTX 5070 için optimize edilmiş
python main.py --epochs 50 --batch 16 --device cuda:0

# Intel iGPU (Linux)
python main.py --epochs 30 --batch 8 --device xpu

# CPU fallback
python main.py --epochs 20 --batch 4 --device cpu
```

## 📊 Dataset Bilgileri

| Model | Images | Classes | Performance |
|-------|---------|---------|-------------|
| Player Detection | ~10K | 1 (Player) | mAP50: 0.85+ |
| Ball Detection | 11,673 | 3 (Player, Ball, Referee) | In development |

## ⚡ Sistem Gereksinimleri

### Minimum
- **CPU:** Intel i5 / AMD Ryzen 5
- **RAM:** 8GB
- **Storage:** 10GB
- **Python:** 3.8+

### Önerilen (Geliştirme)
- **GPU:** RTX 3060+ veya RTX 5070
- **CPU:** Intel i7-12700H+ / AMD Ryzen 7+
- **RAM:** 16GB+
- **Storage:** SSD 50GB+

### Intel iGPU Desteği
- **Minimum:** Intel Iris Xe Graphics
- **OS:** Linux (Intel Extension for PyTorch)
- **Performans:** CPU'ya göre 2-4x hızlı

## 🛠️ Teknoloji Stack

### Machine Learning
- **Framework:** PyTorch
- **Model:** YOLO11n (Ultralytics)
- **Techniques:** Transfer Learning, Multi-class Detection
- **Hardware:** CUDA, Intel iGPU, DirectML, CPU

### Backend
- **Language:** Python/Node.js
- **API:** REST/GraphQL
- **Database:** MongoDB/PostgreSQL

### Frontend
- **Framework:** React/Vue.js
- **UI:** Modern responsive design
- **Real-time:** WebSocket connections

## 📈 Performans Metrikleri

### Training Performance (RTX 5070)
- **Player Detection:** ~5 min/epoch
- **Ball Detection:** ~8 min/epoch
- **Memory Usage:** ~8GB VRAM
- **Inference Speed:** ~10ms/image

### Intel iGPU Performance (i7-12700H)
- **Speedup vs CPU:** 2-4x
- **Ball Detection:** ~20 min/epoch
- **Memory Efficient:** Shared system RAM

## 🔧 Geliştirme Workflow

1. **Dataset hazırlama ve doğrulama**
   ```bash
   python visualize_labels.py
   ```

2. **Model training ve testing**
   ```bash
   python main.py --fraction 0.01 --epochs 5  # Quick test
   python main.py --epochs 50                 # Full training
   ```

3. **Performance analysis**
   ```bash
   # Excel ve HTML raporları otomatik oluşturulur
   ls reports/*/
   ```

## 📁 Proje Yapısı

```
FoMAC/
├── README.md                           # Ana proje dokümantasyonu
├── .gitignore                         # Git ignore (dataset dosyaları hariç)
├── backend/                           # Backend servisleri
│   └── README.md                      # Backend dokümantasyonu
├── frontend/                          # Web arayüzü
│   └── README.md                      # Frontend dokümantasyonu
└── model-training/                    # ML modelleri
    ├── README.md                      # Model training genel bakış
    ├── player-detection/              # Temel oyuncu tespiti
    │   ├── main.py                    # Training scripti
    │   ├── configs/                   # Konfigürasyon dosyaları
    │   └── models/                    # Çıktı modelleri (gitignore)
    └── ball-detection/                # Multi-class detection
        ├── README.md                  # Ball detection dokümantasyonu
        ├── ballDataset/               # Dataset (gitignore)
        │   ├── data.yaml              # YOLO dataset config
        │   ├── images/                # Görüntüler (gitignore)
        │   └── labels/                # Etiketler (gitignore)
        └── yolo/                      # YOLO training pipeline
            ├── README.md              # YOLO pipeline dokümantasyonu
            ├── main.py                # Ana training scripti
            ├── configs/               # Training konfigürasyonları
            ├── modules/               # Core modüller
            ├── utils/                 # Yardımcı fonksiyonlar
            ├── reports/               # Rapor oluşturma
            └── requirements.txt       # Python bağımlılıkları
```

## 🔒 Git Workflow

### Dataset Management
- **Büyük dosyalar gitignore'da:** `ballDataset/images/`, `ballDataset/labels/`
- **Config dosyaları korunur:** `ballDataset/data.yaml`
- **Model weights hariç:** `*.pt`, `*.pth`, `runs/`

### Commit Best Practices
```bash
# Feature branches
git checkout -b feature/ball-detection-optimization

# Model training sonuçları
git add model-training/ball-detection/yolo/
git commit -m "feat: optimize ball detection for RTX 5070"

# Dataset konfigürasyonu
git add ballDataset/data.yaml
git commit -m "config: update ball detection dataset structure"
```

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
# Batch size azalt
python main.py --batch 8 --epochs 30
```

**Poor Ball Detection**
```bash
# Görüntü çözünürlüğünü arttır
python main.py --imgsz 1280 --epochs 50
```

**Intel iGPU Not Working**
```bash
# Intel Extension kur
pip install intel-extension-for-pytorch
```

**Dataset Not Found**
```bash
# ballDataset klasörünü kontrol et
ls ballDataset/data.yaml
```

## 🎯 Roadmap

- [x] Player detection (base model)
- [x] Transfer learning pipeline
- [x] Multi-class detection (Player + Ball + Referee)
- [x] Intel iGPU support
- [ ] Real-time inference API
- [ ] Web dashboard
- [ ] Advanced match analytics
- [ ] Mobile app integration

## 🤝 Contributing

1. Fork the project
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

Bu proje akademik amaçlarla geliştirilmiştir.

## 📞 İletişim

- **Proje:** FoMAC - Football Match Analysis and Classification
- **Geliştirici:** Alperen ARRA
- **Repository:** [arra7-bitirme/FoMAC](https://github.com/arra7-bitirme/FoMAC)