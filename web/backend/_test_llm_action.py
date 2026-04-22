import httpx, json, sys

url = "http://localhost:8001/v1/chat/completions"

# --- 1. connectivity test ---
print("=== 1. Connectivity Test ===")
payload = {
    "model": "nvidia/Qwen3-8B-NVFP4",
    "messages": [
        {"role": "system", "content": "You are a Turkish football commentator."},
        {"role": "user", "content": 'Olay: GOL atildi (confidence: 0.92). Turkce kisa yorum yap. Sadece JSON: {"text": "..."}'},
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "chat_template_kwargs": {"enable_thinking": False},
}
try:
    r = httpx.post(url, json=payload, timeout=30)
    print("status:", r.status_code)
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print("response:", content)
except Exception as e:
    print("ERROR:", e)
    sys.exit(1)

# --- 2. action spotting prompt style test ---
print("\n=== 2. Action Spotting Prompt Style Test ===")
import sys
sys.path.insert(0, r"C:\Users\Admin\Desktop\FoMAC\FoMAC\web\backend")
from pipeline import _build_commentary_item_prompt, _extract_commentary_text_best_effort

fake_item = {
    "t": 18.48,
    "speech_t": 18.48,
    "timecode": "00:18",
    "event_t": 18.48,
    "event_timecode": "00:18",
    "segment_duration_sec": 30.0,
    "window": {"start_t": 0.0, "end_t": 30.0, "duration_sec": 30.0},
    "event_label": "Goal",
    "event_confidence": 0.5986328125,
    "event_source": "action_spotting",
    "description_tr": "Gol oldu!",
    "spotting_model": "SoccerNet_big",
    "spotting_variant": "primary",
}

prompt = _build_commentary_item_prompt(fake_item, [])
print("--- PROMPT (first 400 chars) ---")
print(prompt[:400])
print("--- END PROMPT ---")

payload2 = {
    "model": "nvidia/Qwen3-8B-NVFP4",
    "messages": [
        {"role": "system", "content": "You are a Turkish football commentator."},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.7,
    "max_tokens": 200,
    "chat_template_kwargs": {"enable_thinking": False},
}
try:
    r2 = httpx.post(url, json=payload2, timeout=30)
    print("status:", r2.status_code)
    data2 = r2.json()
    raw2 = data2["choices"][0]["message"]["content"]
    print("raw response:", raw2)
    txt = _extract_commentary_text_best_effort(raw2)
    print("extracted text:", txt)
except Exception as e:
    print("ERROR:", e)

print("\nDONE")
