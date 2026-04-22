"""
test_golsesi.py — golsesi.wav + goal_cheer.wav karışımını test et.

Kullanım:
  python test_golsesi.py                          # 3s ses dosyası üret
  python test_golsesi.py --video path/to/clip.mp4 # videoya göm
  python test_golsesi.py --t 5.0                  # golün 5. saniyede olduğunu simüle et

Çıktı: test_golsesi_output.wav (veya .mp4 video varsa)
"""

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
GOLSESI     = BACKEND_DIR / "golsesi.wav"
CHEER       = BACKEND_DIR / "goal_cheer.wav"

sys.path.insert(0, str(BACKEND_DIR))


def _ffmpeg() -> str:
    """pipeline.py ile aynı yöntemi kullan."""
    try:
        from pipeline import _ffmpeg_exe
        exe = _ffmpeg_exe()
        if exe:
            return exe
    except Exception:
        pass
    sys.exit("ffmpeg bulunamadı (pipeline._ffmpeg_exe başarısız)")


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def test_audio_only(goal_t: float, out_path: Path):
    """golsesi + cheer'i karıştır, saf ses dosyası üret."""
    ff = _ffmpeg()

    golsesi_dur = _wav_duration(GOLSESI)
    cheer_dur   = _wav_duration(CHEER) if CHEER.exists() else 0.0
    total_dur   = goal_t + max(golsesi_dur, cheer_dur) + 1.0   # biraz boşluk

    # Pipeline ile aynı sabit: detection ~1s geç → 2s erken başlat
    _GOAL_SFX_PRE_SEC = 2.0
    sfx_start_t  = max(0.0, goal_t - _GOAL_SFX_PRE_SEC)
    sfx_delay_ms = int(round(sfx_start_t * 1000))

    # golsesi bittikten sonra yorum başlar
    commentary_start_t  = sfx_start_t + golsesi_dur
    commentary_delay_ms = int(round(commentary_start_t * 1000))

    print(f"golsesi.wav süresi : {golsesi_dur:.3f}s")
    print(f"goal_cheer.wav     : {'mevcut' if CHEER.exists() else 'YOK'}")
    print(f"SFX başlangıcı     : {sfx_start_t:.2f}s  (event_t - {_GOAL_SFX_PRE_SEC:.0f}s)")
    print(f"Yorum başlangıcı   : {commentary_start_t:.2f}s  (golsesi bitiminde)")
    print(f"Toplam ses süresi  : {total_dur:.2f}s")
    print()

    inputs = []
    # 1) sessizlik taban
    inputs += ["-f", "lavfi", "-t", f"{total_dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    # 2) golsesi
    inputs += ["-i", str(GOLSESI)]
    # 3) cheer (varsa)
    if CHEER.exists():
        inputs += ["-i", str(CHEER)]

    parts = []
    amix = ["[sil]"]

    parts.append(
        f"[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"atrim=0:{total_dur:.3f},asetpts=N/SR/TB[sil]"
    )

    parts.append(
        f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"adelay={sfx_delay_ms}|{sfx_delay_ms},volume=1.0[gol]"
    )
    amix.append("[gol]")

    if CHEER.exists():
        parts.append(
            f"[2:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"adelay={sfx_delay_ms}|{sfx_delay_ms},volume=0.65[cheer]"
        )
        amix.append("[cheer]")

    n = len(amix)
    parts.append(
        "".join(amix)
        + f"amix=inputs={n}:duration=first:dropout_transition=0:normalize=0[outa]"
    )

    filter_complex = ";".join(parts)

    cmd = [
        ff, "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outa]",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]

    print("FFmpeg komutu:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("HATA:")
        print(result.stderr[-2000:])
        return False

    dur_actual = _wav_duration(out_path)
    print(f"Çıktı      : {out_path}")
    print(f"Çıktı süresi: {dur_actual:.3f}s")
    print()
    print("Beklenen zamanlama:")
    print(f"  {sfx_start_t:.2f}s → golsesi.wav + goal_cheer.wav BAŞLAR")
    print(f"  {commentary_start_t:.2f}s → Burada TTS yorumu başlamalı (pipeline'da)")
    return True


def test_with_video(video_path: str, goal_t: float, out_path: Path):
    """Verilen video klibine golsesi + cheer gömülü test videosu üret."""
    ff = _ffmpeg()

    golsesi_dur = _wav_duration(GOLSESI)
    sfx_start_t  = max(0.0, goal_t - 1.0)
    sfx_delay_ms = int(round(sfx_start_t * 1000))
    commentary_start_t = sfx_start_t + golsesi_dur

    print(f"Video          : {video_path}")
    print(f"SFX başlangıcı : {sfx_start_t:.2f}s")
    print(f"Yorum başlamalı: {commentary_start_t:.2f}s")
    print()

    inputs = ["-i", video_path, "-i", str(GOLSESI)]
    if CHEER.exists():
        inputs += ["-i", str(CHEER)]

    parts = []
    amix = ["[orig]"]

    parts.append(
        "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        "volume=0.3[orig]"
    )
    parts.append(
        f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"adelay={sfx_delay_ms}|{sfx_delay_ms},volume=1.0[gol]"
    )
    amix.append("[gol]")

    if CHEER.exists():
        cheer_idx = 2
        parts.append(
            f"[{cheer_idx}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"adelay={sfx_delay_ms}|{sfx_delay_ms},volume=0.65[cheer]"
        )
        amix.append("[cheer]")

    n = len(amix)
    parts.append(
        "".join(amix)
        + f"amix=inputs={n}:duration=first:dropout_transition=0:normalize=0[outa]"
    )

    cmd = [
        ff, "-y",
        *inputs,
        "-filter_complex", ";".join(parts),
        "-map", "0:v:0",
        "-map", "[outa]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]

    print("FFmpeg komutu:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("HATA:")
        print(result.stderr[-2000:])
        return False

    print(f"Çıktı: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Test videosu (mp4)")
    parser.add_argument("--t", type=float, default=3.0, help="Simüle edilen gol zamanı (saniye)")
    args = parser.parse_args()

    if not GOLSESI.exists():
        sys.exit(f"golsesi.wav bulunamadı: {GOLSESI}")

    print("=" * 60)
    print("golsesi.wav TEST")
    print("=" * 60)
    print()

    if args.video:
        out = BACKEND_DIR / "test_golsesi_output.mp4"
        ok = test_with_video(args.video, args.t, out)
    else:
        out = BACKEND_DIR / "test_golsesi_output.wav"
        ok = test_audio_only(args.t, out)

    if ok:
        print("OK — test tamamlandı")
    else:
        print("BAŞARISIZ")
        sys.exit(1)


if __name__ == "__main__":
    main()
