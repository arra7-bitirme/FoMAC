import json, glob, os
files = sorted(glob.glob('uploads/events_*.json'), key=os.path.getmtime, reverse=True)
if not files:
    print("NO events files found")
    exit()
print("File:", files[0])
with open(files[0], encoding='utf-8') as f:
    d = json.load(f)
print('commentary_error:', d.get('commentary_error'))
print('qwen_vl_events:', json.dumps(d.get('qwen_vl_container_events', []), indent=2)[:800])
print('artifacts keys:', list(d.get('artifacts', {}).keys()))
