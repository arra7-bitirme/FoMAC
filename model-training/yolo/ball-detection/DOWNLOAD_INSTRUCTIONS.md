Kısaca: Hugging Face veri setini indirip `ballDataset` dizinine kaydetme

1) Gerekli paketleri yükleyin (tercihen venv içinde):

```bash
pip install --upgrade pip
pip install datasets huggingface_hub pillow tqdm
```

2) Script'i çalıştırma (varsayılan dataset `Adit-jain/Soccana_player_ball_detection_v1` ve split `train`):

```bash
# ballDataset klasörüne Arrow formatında kaydeder
python model-training/yolo/ball-detection/scripts/download_dataset.py --mode arrow

# Veri setinin örnek yapısını görmek için (hangi alanların bulunduğunu anlamak için)
python model-training/yolo/ball-detection/scripts/download_dataset.py --mode inspect --max-samples 2
```

3) Notlar:
- `--mode arrow` seçeneği `datasets.Dataset.save_to_disk()` kullanarak veri setini Hugging Face Arrow formatında kaydeder. Bu, ileride `datasets.load_from_disk()` ile tekrar yüklemek için uygundur.
- Eğer görüntü dosyalarını (.jpg/.png) ve anotasyon dosyalarını (ör. YOLO formatı) istiyorsanız, önce `--mode inspect` ile örnek kayıt yapısını inceleyin. Görüntü alanının adı muhtemelen `image` veya `img` şeklindedir; anotasyonlar `annotations`, `objects` veya benzeri bir alanda olabilir.
- Arrow formatından dosyaları düz dosya + anotasyon formatına dönüştürme yardımcı script'leri ekleyebilirim; isterseniz, örnek çıktıya göre YOLO formatına dönüştürme script'i hazırlayacağım.

4) Eğer dataset gizli (private) ise: Hugging Face token'ınızı CLI ile ayarlayın veya `huggingface-cli login` ile giriş yapın.

İhtiyacınıza göre ben Arrow kaydetme ve/veya görüntü+anotasyon dışa aktarma (YOLO kuralına göre) script'leri hazırlayabilirim.
