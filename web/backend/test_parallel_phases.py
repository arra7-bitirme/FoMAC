"""
Parallel phase test — two GPU subprocesses running concurrently.
Usage:
    python test_parallel_phases.py                   # GPU stress test only
    python test_parallel_phases.py --video path.mp4  # timing comparison with real pipeline phases
"""
import argparse
import subprocess
import sys
import threading
import time
import tempfile
import os


_GPU_STRESS = """
import sys, time, torch
device = "cuda" if torch.cuda.is_available() else "cpu"
label = sys.argv[1] if len(sys.argv) > 1 else "worker"
if device == "cuda":
    reserved_before = torch.cuda.memory_reserved() / 1e9
    print(f"[{label}] GPU={torch.cuda.get_device_name(0)}  reserved={reserved_before:.2f}GB", flush=True)
x = torch.randn(2048, 2048, device=device)
for i in range(100):
    x = torch.relu(x @ x.T / 2048)
if device == "cuda":
    torch.cuda.synchronize()
    reserved_after = torch.cuda.memory_reserved() / 1e9
    print(f"[{label}] done  reserved={reserved_after:.2f}GB", flush=True)
else:
    print(f"[{label}] done (CPU)", flush=True)
"""


def _stream(proc: subprocess.Popen, label: str, result: dict) -> None:
    lines = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            lines.append(line)
            print(f"  {line}", flush=True)
    except Exception:
        pass
    rc = proc.wait()
    result["rc"] = rc
    result["lines"] = lines


def run_parallel(cmds: list[tuple[str, list[str]]]) -> dict[str, dict]:
    """Launch all cmds simultaneously, stream output with labels, return per-label result dicts."""
    procs = {}
    results = {}
    threads = []
    for label, cmd in cmds:
        results[label] = {"rc": None, "lines": [], "start": time.perf_counter()}
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        procs[label] = p
        t = threading.Thread(target=_stream, args=(p, label, results[label]), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for label, r in results.items():
        r["elapsed"] = time.perf_counter() - r["start"]
    return results


def test_gpu_parallel():
    print("=== GPU parallel stress test ===")
    script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    script.write(_GPU_STRESS)
    script.close()

    try:
        # Sequential baseline
        print("\n-- Sequential --")
        t0 = time.perf_counter()
        for label in ("A", "B"):
            r = run_parallel([(label, [sys.executable, script.name, label])])
            if r[label]["rc"] != 0:
                print(f"FAIL: {label} exit {r[label]['rc']}")
                return False
        seq_time = time.perf_counter() - t0
        print(f"Sequential: {seq_time:.2f}s")

        # Parallel
        print("\n-- Parallel --")
        t0 = time.perf_counter()
        results = run_parallel([
            ("A", [sys.executable, script.name, "A"]),
            ("B", [sys.executable, script.name, "B"]),
        ])
        par_time = time.perf_counter() - t0
        print(f"Parallel: {par_time:.2f}s  (speedup={seq_time/par_time:.2f}x)")

        ok = all(r["rc"] == 0 for r in results.values())
        print(f"\nResult: {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        os.unlink(script.name)


def test_with_video(video_path: str, calib_script: str, tdeed_script: str,
                    calib_cmd: list[str], tdeed_primary_cmd: list[str]):
    """Run calibration + T-DEED primary both sequentially and in parallel, compare timing."""
    print(f"\n=== Real pipeline parallel test: {os.path.basename(video_path)} ===")

    print("\n-- Sequential (calib → tdeed) --")
    t0 = time.perf_counter()
    r_calib_seq = run_parallel([("calib", calib_cmd)])
    r_tdeed_seq = run_parallel([("tdeed", tdeed_primary_cmd)])
    seq_time = time.perf_counter() - t0
    ok_seq = r_calib_seq["calib"]["rc"] == 0 and r_tdeed_seq["tdeed"]["rc"] == 0
    print(f"Sequential: {seq_time:.2f}s  ok={ok_seq}")

    print("\n-- Parallel (calib || tdeed) --")
    t0 = time.perf_counter()
    results = run_parallel([
        ("calib", calib_cmd),
        ("tdeed", tdeed_primary_cmd),
    ])
    par_time = time.perf_counter() - t0
    ok_par = all(r["rc"] == 0 for r in results.values())
    print(f"Parallel: {par_time:.2f}s  ok={ok_par}  speedup={seq_time/par_time:.2f}x")

    return ok_par


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="", help="video path for real pipeline test")
    args = ap.parse_args()

    ok = test_gpu_parallel()

    if args.video and os.path.isfile(args.video):
        print("\nSkipping real pipeline test (provide --calib_cmd and --tdeed_cmd manually)")

    sys.exit(0 if ok else 1)
