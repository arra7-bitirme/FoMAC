#!/usr/bin/env bash
set -euo pipefail

# dev.sh — run backend (background) + frontend (foreground) in one terminal
# Kills existing servers on ports 8000, 3000, 3001 then starts backend and frontend.

echo "Stopping any processes on ports 8000, 3000, 3001..."
# fuser may not exist; try ss + awk fallback
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true
fuser -k 3001/tcp 2>/dev/null || true
sleep 1

# Show listeners
ss -ltnp | grep -E ":8000|:3000|:3001" || true

echo "Starting backend in background (logs -> backend/backend.log)"
cd backend
# ensure uploads dir exists
mkdir -p uploads
# Start uvicorn without reload to keep single process
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info > ../backend.log 2>&1 &
BACKEND_PID=$!
cd - >/dev/null
sleep 1

echo "Backend started (pid=$BACKEND_PID). Logs: backend.log"

echo "Starting frontend (Next.js) in foreground"
# Start frontend in foreground so user sees logs; use npm run dev
npm run dev

# When frontend exits, kill backend
echo "Frontend exited; stopping backend (pid=$BACKEND_PID)"
kill $BACKEND_PID 2>/dev/null || true
wait $BACKEND_PID 2>/dev/null || true

echo "Stopped."