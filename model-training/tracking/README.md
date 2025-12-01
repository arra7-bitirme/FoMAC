# Player Tracking & Re-Identification

Bu modül, **SoccerNet ReID** dataset'i ve **SNMOT** formatındaki tracking dataset'lerini kullanarak re-identification destekli çoklu oyuncu takibini FoMAC projesine entegre eder. Pipeline, mevcut **player/ball/referee** YOLO modelinizin (`models/player_ball_detector/weights/best.pt`) çıktıları üzerinde inşa edilir.

## 🚧 Mimari Özeti

- **Algı (Detection):** `best.pt` modeli Ultralytics YOLO arayüzü ile her karede oyuncu/top/hakem tespiti üretir.
- **Özellik Öğrenimi (ReID):** SoccerNet ReID dataset'inden türetilen özel bir ResNet tabanlı embedder ile oyuncu kimlik vektörleri üretilir.
- **Takip:** Sınıf farkındalığı olan IoU tabanlı bir tracker (oyuncular, hakemler, top için ayrı kanallar) ile ID eşleşmesi yapılır, MOTChallenge + CSV formatında sonuçlar yazılır. (İsteğe bağlı olarak StrongSORT/ByteTrack config'leri ileride bağlanabilir.)

```
Frames -> YOLO best.pt -> Detections -> StrongSORT (+ReID embeddings) -> Tracks (.txt)
                             ↑
                      train_reid.py (SoccerNet)
```

## 📦 Bağımlılıklar

```powershell
cd model-training/tracking
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Başlıca paketler: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `lap`, `filterpy`, `scipy`, `scikit-learn`, `motmetrics`, `pyyaml`.

## 🗂️ Dataset Hazırlığı

### 1. Tracking Dataset (SNMOT / MOT formatı)

1. Tüm SNMOT sekansları `c:\Users\kaan.aggunlu\Desktop\git\FoMAC\tracking` altında zaten `train/` ve `test/` klasörlerine ayrılmış durumda (örnek: `train/SNMOT-060`, `test/SNMOT-197`).
2. Her sekans aşağıdaki yapıya sahip olmalıdır:
   ```
   SNMOT-060/
     ├─ seqinfo.ini
     ├─ gameinfo.ini
     ├─ img1/
     ├─ det/det.txt (opsiyonel)
     └─ gt/gt.txt  (opsiyonel)
   ```
3. `configs/tracking.yaml` içindeki `datasets.root` alanını bu dizine (ya da eşdeğerine) işaret edecek şekilde güncelleyin ve `datasets.splits` listesi ile hangi alt klasörlerin otomatik taranacağını belirtin (varsayılan: `train` + `test`).
4. `datasets.sequences` alanını `null` bırakırsanız belirtilen split'lerdeki **tüm** sekanslar otomatik olarak işlenecektir.

> Not: Train klasörü ağırlıklı olarak `SNMOT-060`–`SNMOT-077` ve `SNMOT-097`–`SNMOT-170` dizilerini içeriyor; test klasörü ise `SNMOT-116`–`SNMOT-200` aralığındaki ~70 maç klibini barındırıyor. Bu yapı MOTChallenge formatına uyumlu olduğu sürece direkt kullanılabilir.

### 2. SoccerNet ReID Dataset'i

1. [sn-reid](https://github.com/SoccerNet/sn-reid) yönergelerini izleyerek `train/ valid/ test/ challenge` klasörlerini indirin.
2. Bu klasörleri şu yapıda tutun:
   ```
   <datasets_root>/soccernetv3/reid/{train,valid,test,challenge}
   ```
3. `configs/reid.yaml` içindeki `reid_dataset.root` alanını bu yol ile değiştirin.

## ⚙️ Konfigürasyon Dosyaları

| Dosya | Amaç |
|-------|------|
| `configs/reid.yaml` | ReID eğitimi için dataset yolları, hiperparametreler |
| `configs/tracking.yaml` | YOLO tracking, sınıf grupları, team-ID ve CSV/visualization ayarları |
| `configs/trackers/strongsort_soccer.yaml` | StrongSORT ayarları + ReID ağı ağırlıkları |

Örnek `team_classification` bloğu:

```yaml
team_classification:
   enabled: true
   method: color
   samples_per_track: 12
   min_track_hits: 6
   color_space: lab
   clusters: 2

tracking:
   reid:
      enabled: true
      weights: outputs/reid/checkpoints/best_reid.pt
      alpha: 0.6  # IoU weight
      beta: 0.4   # embedding distance weight
      max_distance: 0.7
      image_size: [256, 128]
```

## 🧠 ReID Eğitimi

```powershell
cd model-training/tracking
python train_reid.py --config configs/reid.yaml
```

- Çıktılar `outputs/reid/` altında saklanır (`best_reid.pt`).
- Config üzerinden `epochs`, `batch_size`, `image_size` gibi değerleri değiştirebilirsiniz.
- `configs/reid.yaml` içindeki `feature_extractor` bölümünde hangi YOLO ağırlığının (örn. `models/player_ball_detector/weights/best.pt`) kullanılacağını ve hangi neck katmanının dondurulacağını seçersiniz. Bu blok varsayılan olarak dedektörün `head.f` listesindeki son feature map'i alır ve tüm dedektörü freeze eder.
- `embedding_head` bölümü, backbone'dan çıkan vektörün kaç boyuta projeleneceğini ve özdeşlik (L2 normalize) çıkışının aktifleştirilip aktifleştirilmeyeceğini kontrol eder. Eğitim sırasında sadece bu başlık güncellenir; detection mAP sabit kalır.

## 🎯 Tracking Çalıştırma

```powershell
cd model-training/tracking
python run_tracking.py --config configs/tracking.yaml --sequence SNMOT-060 --save-visuals --team-color
```

- YOLO modeli `models/player_ball_detector/weights/best.pt` dosyasından yüklenir.
- Her sekans için `outputs/tracks/<sequence>/` klasörü açılır; içinde hem MOT (`<sequence>.txt`) hem de genişletilmiş CSV (`<sequence>.csv`) dosyaları saklanır. CSV; `frame,track_id,class_id,x,y,w,h,score,team_id` kolonlarını içerir ve sonraki fazlarda (team-ID, istatistik) doğrudan kullanılır.
- `tracking.save_visualizations = true` veya CLI `--save-visuals` parametresi ile `outputs/tracks/<sequence>/visualizations/` altında ID annotate kareler kaydedilir.
- `output.write_video = true` diyerek bu karelerden otomatik olarak `outputs/tracks/<sequence>/video/<sequence>_annotated.mp4` üretilebilir.
- Kalabalık sahnelerde ID karışmasını azaltmak için `tracking.appearance_embedding` bloğu varsayılan olarak Lab renk histogramı çıkarıp IoU + renk mesafesi ile eşleşmeleri güçlendirir. `alpha/beta/max_distance` değerleriyle IoU/renk dengesini ayarlayabilirsiniz.
- `team_classification.enabled = true` olduğunda K-Means ile iki takım kümesi çıkartılır ve hem CSV `team_id` kolonuna hem de annotate video renklerine (T0/T1 etiketi) yansıtılır.
- CLI boyunca ilerlemeyi görmek için `tracking.progress = true` bırakın; her sekans Rich tabanlı bir loading bar ile toplam kare sayısında nereye gelindiğini gösterir.
- Harici maç videolarını sadece config üzerinden çalıştırmak isterseniz `configs/tracking.yaml` içine şu bloğu ekleyin:
   ```yaml
   videos:
      sources:
         - C:\videos\match01.mp4
      names:
         - chelsea_swans
   datasets:
      enabled: false  # sadece video işlensin, SNMOT klasörleri atlanır
   ```
   Ardından yalnızca `python run_tracking.py --config configs/tracking.yaml` komutu yeterlidir.
- CLI üzerinden `--team-color`, `--team-method`, `--team-samples`, `--team-min-hits`, `--no-team` bayrakları ile home/away sınıflandırması kontrol edilebilir.
- ReID entegrasyonunu test etmek için `--enable-reid`, `--reid-weights`, `--reid-alpha`, `--reid-beta`, `--reid-max-distance` parametreleri kullanılabilir. Varsayılan olarak devre dışı gelir.
- CLI üzerinden `--player-classes`, `--ball-classes`, `--imgsz`, `--conf`, `--iou`, `--vid-stride`, `--no-csv` gibi parametrelerle pipeline sahaya göre hızlıca ayarlanabilir.

## 📁 Klasör Yapısı

```
tracking/
├── configs/
│   ├── reid.yaml
│   ├── tracking.yaml
│   └── trackers/strongsort_soccer.yaml
├── modules/
│   ├── __init__.py
│   ├── class_aware_tracker.py
│   ├── datasets.py
│   ├── detector_stream.py
│   ├── reid_training.py
│   ├── tracker_runner.py
│   └── visualization.py
├── requirements.txt
├── run_tracking.py
└── train_reid.py
```

## ✅ Gelecek Adımlar

- `tools/evaluate_mot.py`: GT ile IDF1/IDSW hesaplamak için eklenebilir.
- Daha hafif ReID mimarileri (EfficientNet tabanlı) eklenip `configs/reid.yaml` üzerinden seçilebilir.
- Çoklu GPU / DDP desteği için `train_reid.py` genişletilebilir.
- **Team Classification (Phase 2):** `configs/tracking.yaml` içindeki `team_classification` bloğunu aktifleştirip (`enabled: true`, `method: color`) veya CLI `python run_tracking.py ... --team-color` komutu ile çalıştırın. Sistem her oyuncu track'i için forma renklerini örnekleyip K-Means ile 2 küme oluşturur. Sonuçlar CSV'deki `team_id` kolonuna yazılır ve ilerideki istatistik/rapor modülleri tarafından doğrudan kullanılabilir.
- **ReID Karşılaştırması (Phase 3):** `python scripts/compare_reid.py --config configs/tracking.yaml --sequence SNMOT-060 --gt path/to/gt/gt.txt` komutu aynı sekans için ReID açık/kapalı sonuçlarını üretir ve `motmetrics` üzerinden IDF1, IDSW gibi metrikleri raporlar.

Tüm pipeline tek config üzerinden yönetilebilir durumdadır; varsayılan ayarlar FoMAC içindeki `best.pt` ağırlığını ve örnek dataset yollarını referans alır.
