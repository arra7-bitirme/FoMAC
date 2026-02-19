Frontend

Proje
Bu klasör Next.js tabanlı frontend uygulamasını içerir. Uygulama `app/`, `components/` ve `lib/` gibi dizinler altında düzenlenmiştir.

Gereksinimler
- Node.js 16 veya daha yeni (tercihen Node 18 LTS)
- `npm` veya uyumlu paket yöneticisi

Kurulum
1. Proje dizinine girin:

   cd frontend

2. Bağımlılıkları yükleyin:

   npm install

Çalıştırma (geliştirme)

  npm run dev

Alternatif olarak `dev.sh` betiğiniz varsa çalıştırabilirsiniz:

  chmod +x dev.sh
  ./dev.sh

Üretim için yapı

  npm run build
  npm start

Dosya Yapısı (kısa)
- `app/`: Next.js sayfa ve route yapısı.
- `components/`: Yeniden kullanılabilir React bileşenleri.
- `lib/`: Yardımcı fonksiyonlar ve durum yönetimi (`store.ts`, `utils.ts`).
- `components/ui/`: UI yardımcı bileşenleri (buton, input, vb.).

Çevresel Değişkenler
- Ortam değişkenlerini `env.local` veya platformunuza uygun şekilde tanımlayın. Next.js çalıştırma sırasında bunlar okunur.

Geliştirme
- Kod değişikliklerinden sonra otomatik yeniden yükleme çalışmalıdır.
- Stil için Tailwind yapılandırması `tailwind.config.ts` içinde mevcuttur.

Sorun Giderme
- Bağımlılık veya derleme hatası alırsanız `node` sürümünü kontrol edin ve `npm install` komutunu yeniden çalıştırın.

Katkıda Bulunma
- Yeni özellikler ve hata düzeltmeleri için issue açın veya pull request gönderin.
