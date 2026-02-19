Backend

Proje
Bu klasör uygulamanın Python tabanlı backend bileşenini içerir. Ana giriş noktası `main.py` dosyasıdır; çalıştırma için ayrıca `run.sh` ve bağımlılıklar `requirements.txt` içinde tanımlanmıştır.

Gereksinimler
- Python 3.8 veya daha yeni
- `pip` ve sanal ortam aracı (venv)

Kurulum
1. Proje dizinine girin:

   cd backend

2. Sanal ortam oluşturup etkinleştirin:

   python -m venv .venv
   source .venv/bin/activate

3. Bağımlılıkları yükleyin:

   pip install -r requirements.txt

Çalıştırma
- Hazır betik ile (izin verildiyse):

  chmod +x run.sh
  ./run.sh

- Alternatif olarak doğrudan Python ile:

  python main.py

Dosyalar
- `main.py`: Uygulama giriş noktası.
- `requirements.txt`: Python bağımlılıkları.
- `run.sh`: Ortamı hazırlayıp sunucuyu başlatmaya yardımcı betik.
- `uploads/`: Uygulamanın yüklenen verileri sakladığı dizin (örnek JSON dosyaları içerir).

Geliştirme
- Değişiklik yapmadan önce sanal ortamı etkinleştirin.
- Geliştirme sırasında çıktı ve hatalar `main.py` tarafından konsola yazdırılır; loglama eklemek isterseniz mevcut kodu düzenleyin.

Sorun Giderme
- Bağımlılık hatası alıyorsanız sanal ortamın etkin olduğundan ve `pip install -r requirements.txt` komutunun başarılı tamamlandığından emin olun.
- Port çakışması varsa çalışan başka bir süreç yoksa farklı bir port belirleyin veya mevcut süreci sonlandırın.

Katkıda Bulunma
- Küçük düzeltmeler ve hata düzeltmeleri için issue açın veya pull request gönderin.

İletişim
- Proje ile ilgili sorular için depo sahibiyle iletişime geçin.
