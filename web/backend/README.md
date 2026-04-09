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
   ./run.sh start

   Durum/log:
   ./run.sh status
   ./run.sh logs
   ./run.sh follow
   ./run.sh stop

- Config dosyası ile (önerilen):

   ```bash
   # örnek config: run.env.example
   ./run.sh start --config ./run.env.example
   ```

   Not: config dosyası `KEY=VALUE` formatında bir env dosyasıdır.
   En sık kullanılanlar: `FOMAC_CONDA_ENV`, `FOMAC_YOLO_MODEL_PATH`, `FOMAC_VIDEO_DIR`.

- Windows (PowerShell) ile:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run.ps1 start  -Config .\run.env.example
   powershell -ExecutionPolicy Bypass -File .\run.ps1 status -Config .\run.env.example
   powershell -ExecutionPolicy Bypass -File .\run.ps1 logs   -Config .\run.env.example
   powershell -ExecutionPolicy Bypass -File .\run.ps1 follow -Config .\run.env.example
   powershell -ExecutionPolicy Bypass -File .\run.ps1 stop   -Config .\run.env.example
   ```

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

Commentary TTS (XTTS v2)
- Commentary text generation now defaults to Ollama:

   - `commentary_llm_backend=ollama`
   - `commentary_llm_url=http://localhost:11434/`
   - `commentary_llm_model=qwen3.5:9b`

- Jersey number recognition remains on the separate Qwen-VL server (`qwen_vl_url`, default `http://localhost:8080/`).
- The backend can now manage the Qwen-VL Docker container directly:

   - `qwen_vl_manage_container=true`
   - `qwen_vl_container_id=4d2ba276bce6347e95bb962a538bf43d70057a151d0ea03e35110c85ec0ec36c`
   - `qwen_vl_stop_before_commentary=true`
   - `qwen_vl_ready_timeout_sec=60.0`

- During a pipeline run, the backend starts that container before jersey inference and stops it before commentary generation.
- Commentary generation will not call Ollama unless the VL container stop is confirmed.
- Commentary prompts now include richer match-state context built from calibration frame windows, possession changes, calibration events, and jersey-aware nearby-player summaries.
- Tunable request fields:

   - `commentary_context_window_sec=12.0`
   - `commentary_context_stride_sec=1.0`
   - `commentary_context_max_samples=9`
   - `commentary_segment_sec=30.0`
   - `commentary_state_interval_sec=10.0`
   - `commentary_llm_timeout_sec=90.0`
   - `commentary_min_audio_gap_sec=0.35`

- Commentary audio is no longer limited to action spotting only. The backend now builds commentary anchors from action spotting, calibration events, possession changes, and periodic state windows across the match timeline.
- Commentary is now produced as one short clip per timeline segment by default, so each speech line stays inside its own window instead of running across the next event block.
- LLM generation runs per commentary item instead of one large batch, which reduces Ollama timeout risk and makes it easier to synthesize audio for all commentary entries.
- Recent commentary lines are fed back into the prompt to reduce repetition, and synthesized audio clips are scheduled with a minimum gap to avoid overlapping speech.
- Before commentary generation, the pipeline performs a best-effort local CUDA cache cleanup so Ollama can claim GPU memory more reliably.
- Varsayılan backend: `xttsv2` (Coqui XTTS v2).
- XTTS v2, bir referans ses dosyası ister. Windows'ta env ile verin:

   - `COMMENTARY_SPEAKER_WAV=C:\\path\\to\\speaker.wav`
   - (opsiyonel) `COMMENTARY_TTS_BACKEND=xttsv2` (veya `sapi` / `pyttsx3`)

- XTTS kurulumu için: `pip install -r requirements.txt` (gerekirse ayrıca uygun `torch` wheel'i kurmanız gerekebilir).
- Eğer `ModuleNotFoundError: No module named 'pkg_resources'` görürseniz, `setuptools==70.3.0` pin'i gereklidir (requirements içinde var).

Katkıda Bulunma
- Küçük düzeltmeler ve hata düzeltmeleri için issue açın veya pull request gönderin.

İletişim
- Proje ile ilgili sorular için depo sahibiyle iletişime geçin.
