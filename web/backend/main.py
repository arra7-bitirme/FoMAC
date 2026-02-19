
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import cv2
import base64
import time
import numpy as np
from typing import Optional, List
import logging
import json
from contextlib import asynccontextmanager

from fastapi.responses import FileResponse
import mimetypes
import subprocess
import hashlib
import shutil
from fastapi.responses import StreamingResponse, Response
from datetime import datetime
from urllib.parse import quote as requests_quote
import random

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Configuration ---
YOLO_MODEL_PATH = "/home/alperen/Downloads/best.pt"

# This dictionary will hold the loaded model
ml_models = {}

# --- Lifespan Management (for loading models on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the YOLO model
    try:
        from ultralytics import YOLO
        if os.path.exists(YOLO_MODEL_PATH):
            logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
            ml_models["yolo"] = YOLO(YOLO_MODEL_PATH)
            logger.info("YOLO model loaded successfully")
        else:
            logger.warning(f"YOLO model not found at {YOLO_MODEL_PATH}. Inference will not be available.")
            ml_models["yolo"] = None
    except ImportError:
        logger.warning("ultralytics not installed. YOLO inference will be unavailable.")
        ml_models["yolo"] = None
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        ml_models["yolo"] = None
    
    yield
    
    # Clean up the ML models and release the resources
    ml_models.clear()
    logger.info("ML models cleaned up.")


app = FastAPI(title="FOMAC Backend", version="1.1.0", lifespan=lifespan)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    # include both 3000 and 3001 (Next.js may pick the next free port)
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models for Request/Response ---
class FrameRequest(BaseModel):
    frame_id: float
    frame_data: str # base64 encoded image string

class Detection(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str

class FrameResponse(BaseModel):
    frame_id: float
    inference_time: float
    detections: List[Detection]


# --- API Endpoints ---

# --- Debug Video Generation ---
from typing import Dict
class ProcessVideoRequest(BaseModel):
    path: str
    minutes: Optional[float] = None
    start_seconds: Optional[float] = 0.0

@app.post("/api/process_video")
async def process_video(req: ProcessVideoRequest):
    """
    For each request, generate two videos:
    - Normal video (just extract/copy the segment)
    - YOLO-debug video (bounding boxes drawn)
    Returns URLs for both.
    """
    requested_path = os.path.abspath(req.path)
    safe_base_dir = os.path.abspath(VIDEO_DIR)
    if not (requested_path.startswith(safe_base_dir) or requested_path.startswith(UPLOADS_DIR)) or not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")

    duration = None
    if req.minutes is not None:
        duration = float(req.minutes) * 60.0
    start = float(req.start_seconds or 0.0)

    base_name = os.path.splitext(os.path.basename(requested_path))[0]
    # Generate a 32-bit unique id for this processed video
    video_id = random.getrandbits(32)
    id_hex = f"{video_id:08x}"
    normal_name = f"{base_name}_{id_hex}_normal.mp4"
    debug_name = f"{base_name}_{id_hex}_debug.mp4"
    normal_path = os.path.join(UPLOADS_DIR, normal_name)
    debug_path = os.path.join(UPLOADS_DIR, debug_name)

    # --- Normal video: extract/copy segment ---
    ffmpeg_bin = shutil.which("ffmpeg")
    src_fps = None
    src_width = None
    src_height = None
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", str(start),
            "-i", requested_path,
            "-t", str(duration) if duration else "",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            normal_path,
        ]
        # Remove empty args
        cmd = [c for c in cmd if c != ""]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed for normal video: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"ffmpeg failed for normal video: {e.stderr}")
    else:
        # Fallback: OpenCV extraction
        cap = cv2.VideoCapture(requested_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open video for extraction")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        start_frame = int(start * src_fps)
        total_frames = int(duration * src_fps) if duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(normal_path, fourcc, src_fps, (src_width, src_height))
        if not writer.isOpened():
            cap.release()
            raise HTTPException(status_code=500, detail="Failed to create video writer for extraction")
        written = 0
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != src_width or frame.shape[0] != src_height:
                frame = cv2.resize(frame, (src_width, src_height))
            writer.write(frame)
            written += 1
        writer.release()
        cap.release()
        if written == 0:
            try:
                os.remove(normal_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail="No frames written during extraction")

    # We will not create a separate debug video file here. The frontend treats
    # debug vs normal: create a separate debug video with YOLO boxes if model available.
    # Ensure the normal video exists and is non-empty, then optionally produce debug outputs.
    if not os.path.isfile(normal_path) or os.path.getsize(normal_path) == 0:
        logger.error(f"Normal video missing after processing: {normal_path}")
        raise HTTPException(status_code=500, detail="Normal video not produced")

    # Prepare debug outputs
    debug_json_path = os.path.join(UPLOADS_DIR, f"{base_name}_{id_hex}_debug.json")
    try:
        yolo_model = ml_models.get("yolo")
    except Exception:
        yolo_model = None

    if yolo_model is not None:
        try:
            cap = cv2.VideoCapture(normal_path)
            if not cap.isOpened():
                raise Exception("Failed to open normal video for debug generation")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

            temp_debug = normal_path + ".tmp_debug.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_debug, fourcc, fps, (w, h))
            frame_idx = 0
            detections_list = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # run inference on RGB frame
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    frame_rgb = frame
                try:
                    results = yolo_model(frame_rgb, verbose=False, imgsz=640, conf=0.25)
                except TypeError:
                    results = yolo_model.predict(frame_rgb, imgsz=640, conf=0.25)

                img_h, img_w = frame.shape[:2]
                frame_dets = []
                inference_time_ms = 0.0
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = yolo_model.names.get(class_id, "unknown").lower()
                        if class_name in ["player", "ball", "referee", "person"]:
                            det = {
                                "x": float(x1 / img_w),
                                "y": float(y1 / img_h),
                                "width": float((x2 - x1) / img_w),
                                "height": float((y2 - y1) / img_h),
                                "confidence": float(box.conf[0].cpu().numpy()),
                                "class_name": "player" if class_name == "person" else class_name,
                            }
                            frame_dets.append(det)
                            # draw on frame (BGR)
                            x = int(det["x"] * img_w)
                            y = int(det["y"] * img_h)
                            w_box = int(det["width"] * img_w)
                            h_box = int(det["height"] * img_h)
                            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                            label = f"{det['class_name']} {int(det['confidence']*100)}%"
                            cv2.putText(frame, label, (x, max(10, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                # timestamp in seconds
                timestamp = frame_idx / fps
                detections_list.append({"frame_id": timestamp, "inference_time": inference_time_ms, "detections": frame_dets, "received_time": time.time()})
                writer.write(frame)
                frame_idx += 1

            writer.release()
            cap.release()

            # Re-encode temp_debug to ensure libx264 compatibility if ffmpeg exists
            if shutil.which("ffmpeg"):
                ffmpeg_bin = shutil.which("ffmpeg")
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    temp_debug,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    debug_path,
                ]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"ffmpeg re-encode failed for debug video: {e.stderr}")
                    # Fallback: move temp_debug to debug_path
                    try:
                        shutil.move(temp_debug, debug_path)
                    except Exception:
                        pass
                finally:
                    try:
                        if os.path.exists(temp_debug):
                            os.remove(temp_debug)
                    except Exception:
                        pass
            else:
                try:
                    shutil.move(temp_debug, debug_path)
                except Exception:
                    pass

            # write detections JSON
            try:
                with open(debug_json_path, "w") as jf:
                    json.dump(detections_list, jf)
            except Exception as e:
                logger.error(f"Failed to write debug json: {e}")

        except Exception as e:
            logger.error(f"Failed to create debug video: {e}")
            # If debug generation fails, fall back to using normal as debug
            debug_path = normal_path
            debug_json_path = ""
    else:
        # No model available, point debug to normal
        debug_path = normal_path
        debug_json_path = ""

    debug_json_url = f"http://localhost:8000/api/file?path={requests_quote(debug_json_path)}" if debug_json_path else ""

    return {
        "video_id": video_id,
        "normal_path": normal_path,
        "normal_url": f"http://localhost:8000/api/video?path={requests_quote(normal_path)}",
        "debug_path": debug_path,
        "debug_url": f"http://localhost:8000/api/video?path={requests_quote(debug_path)}",
        "debug_json_path": debug_json_path,
        "debug_json_url": debug_json_url,
    }
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "FOMAC Backend"}

@app.get("/health")
async def health():
    """Health check with model status"""
    model_loaded = ml_models.get("yolo") is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": YOLO_MODEL_PATH if model_loaded else None,
    }


@app.get("/api/model_info")
async def model_info():
    yolo = ml_models.get("yolo")
    if yolo is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    try:
        names = getattr(yolo, 'names', None)
        return {"names": names}
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Video Discovery ---
VIDEO_DIR = "/home/alperen/bitirme/soccerNet/england_epl"
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}
UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads"))

# Ensure uploads directory exists and is writable
os.makedirs(UPLOADS_DIR, exist_ok=True)

class VideoFile(BaseModel):
    path: str
    name: str

@app.get("/api/videos", response_model=List[VideoFile])
async def list_videos():
    """
    Recursively scans the video directory and returns a list of supported video files.
    """
    video_files = []
    if not os.path.isdir(VIDEO_DIR):
        logger.warning(f"Video directory not found: {VIDEO_DIR}")
        return []
        
    for root, _, files in os.walk(VIDEO_DIR):
        for file in sorted(files):
            if any(file.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS):
                full_path = os.path.join(root, file)
                video_files.append(VideoFile(path=full_path, name=file))
    
    logger.info(f"Found {len(video_files)} videos in {VIDEO_DIR}")
    return video_files


@app.get("/api/video")
async def get_video_stream(request: Request, path: str):
    """
    Streams a video file from the filesystem.
    Takes a 'path' query parameter, which should be a valid path to a video file.
    """
    # Security: Ensure the requested path is within the allowed video directory
    # and is a real file, to prevent directory traversal attacks.
    safe_base_dir = os.path.abspath(VIDEO_DIR)
    requested_path = os.path.abspath(path)
    # Allow serving from either VIDEO_DIR or the backend uploads directory
    allowed = (
        requested_path.startswith(safe_base_dir) or requested_path.startswith(UPLOADS_DIR)
    ) and os.path.isfile(requested_path)

    if not allowed:
        raise HTTPException(status_code=404, detail="Video not found or access denied")
    
    # Determine media type (ensure mp4 is recognized)
    media_type, _ = mimetypes.guess_type(requested_path)
    if not media_type:
        if requested_path.lower().endswith('.mp4'):
            media_type = 'video/mp4'
        else:
            media_type = 'application/octet-stream'

    # Support HTTP Range requests for video playback in browsers
    range_header = request.headers.get('range')
    if not range_header:
        return FileResponse(requested_path, media_type=media_type)

    # Parse range header (e.g., 'bytes=0-')
    size = os.path.getsize(requested_path)
    try:
        units, range_part = range_header.split('=')
        start_str, end_str = range_part.split('-')
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else size - 1
    except Exception:
        # Malformed range; return whole file
        return FileResponse(requested_path, media_type=media_type)

    if start >= size:
        raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

    end = min(end, size - 1)
    length = end - start + 1

    def stream_range(path, start, length):
        with open(path, 'rb') as f:
            f.seek(start)
            remaining = length
            chunk_size = 1024 * 1024
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        'Content-Range': f'bytes {start}-{end}/{size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(length),
        'Content-Type': media_type,
    }
    return StreamingResponse(stream_range(requested_path, start, length), status_code=206, headers=headers)


@app.get("/api/uploads")
async def list_uploaded_videos():
    files = []
    if not os.path.isdir(UPLOADS_DIR):
        return files
    for f in sorted(os.listdir(UPLOADS_DIR)):
        full = os.path.join(UPLOADS_DIR, f)
        if not (os.path.isfile(full) and any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)):
            continue
        # Try to extract video_id from filename pattern: base_<hex8>_normal.mp4
        video_id = None
        try:
            name_parts = f.rsplit(".", 1)[0].split("_")
            # look for 8-hex chunk in parts
            for part in name_parts:
                if len(part) == 8:
                    try:
                        video_id = int(part, 16)
                        break
                    except ValueError:
                        continue
        except Exception:
            video_id = None
        item = VideoFile(path=full, name=f)
        # attach video_id if found (pydantic model will ignore extras on response)
        files.append({"path": item.path, "name": item.name, **({"video_id": video_id} if video_id is not None else {})})
    return files


@app.get("/api/uploaded_video")
async def get_uploaded_video(path: str):
    requested_path = os.path.abspath(path)
    if not requested_path.startswith(UPLOADS_DIR) or not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Uploaded video not found or access denied")
    media_type, _ = mimetypes.guess_type(requested_path)
    if not media_type or not media_type.startswith("video/"):
        media_type = "application/octet-stream"
    return FileResponse(requested_path, media_type=media_type)


@app.get("/api/file")
async def get_file(path: str):
    requested_path = os.path.abspath(path)
    safe_base_dir = os.path.abspath(VIDEO_DIR)
    # Allow files from UPLOADS_DIR or VIDEO_DIR
    if not (requested_path.startswith(safe_base_dir) or requested_path.startswith(UPLOADS_DIR)) or not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="File not found or access denied")
    media_type, _ = mimetypes.guess_type(requested_path)
    if not media_type:
        media_type = "application/octet-stream"
    return FileResponse(requested_path, media_type=media_type)


@app.get("/api/thumbnail")
async def get_thumbnail(path: str):
    requested_path = os.path.abspath(path)
    safe_base_dir = os.path.abspath(VIDEO_DIR)
    if not (requested_path.startswith(safe_base_dir) or requested_path.startswith(UPLOADS_DIR)) or not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")
    try:
        cap = cv2.VideoCapture(requested_path)
        success, frame = cap.read()
        cap.release()
        if not success or frame is None:
            raise HTTPException(status_code=500, detail="Failed to read frame for thumbnail")
        # encode as jpeg
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode thumbnail")
        return Response(content=buf.tobytes(), media_type='image/jpeg')
    except Exception as e:
        logger.error(f"Error generating thumbnail for {requested_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ExtractRequest(BaseModel):
    path: str
    minutes: float
    start_seconds: Optional[float] = 0.0


@app.post("/api/extract")
async def extract_clip(req: ExtractRequest):
    requested_path = os.path.abspath(req.path)
    safe_base_dir = os.path.abspath(VIDEO_DIR)
    if not (requested_path.startswith(safe_base_dir) or requested_path.startswith(UPLOADS_DIR)) or not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Video not found or access denied")

    # ensure uploads dir exists
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    duration = float(req.minutes) * 60.0
    start = float(req.start_seconds or 0.0)

    base_name = os.path.splitext(os.path.basename(requested_path))[0]
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    out_name = f"{base_name}_clip_{timestamp}.mp4"
    out_path = os.path.join(UPLOADS_DIR, out_name)

    # Try to use ffmpeg if available, otherwise fall back to OpenCV-based extraction
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        # Build ffmpeg command (re-encode to ensure compatibility)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            str(start),
            "-i",
            requested_path,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            out_path,
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e.stderr}")
    else:
        # Fallback: use OpenCV to read frames and write a new MP4
        try:
            cap = cv2.VideoCapture(requested_path)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for extraction")

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

            start_frame = int(start * src_fps)
            total_frames = int(duration * src_fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, src_fps, (src_width, src_height))
            if not writer.isOpened():
                cap.release()
                raise HTTPException(status_code=500, detail="Failed to create video writer for extraction")

            written = 0
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # ensure correct size
                if frame.shape[1] != src_width or frame.shape[0] != src_height:
                    frame = cv2.resize(frame, (src_width, src_height))
                writer.write(frame)
                written += 1

            writer.release()
            cap.release()

            if written == 0:
                # remove zero-length file if created
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail="No frames written during extraction")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"OpenCV extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"OpenCV extraction failed: {e}")

    # verify output exists and is non-empty
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        logger.error(f"Extraction produced no output file: {out_path}")
        raise HTTPException(status_code=500, detail="Extraction failed to produce output file")

    return {"path": out_path, "url": f"http://localhost:8000/api/video?path={requests_quote(out_path)}"}


class DeleteRequest(BaseModel):
    path: str


@app.post("/api/delete_upload")
async def delete_upload(req: DeleteRequest):
    requested_path = os.path.abspath(req.path)
    if not requested_path.startswith(UPLOADS_DIR) or not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found or access denied")
    try:
        os.remove(requested_path)
        return {"deleted": True}
    except Exception as e:
        logger.error(f"Failed to delete uploaded file {requested_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process_frame", response_model=FrameResponse)
async def process_frame(request: FrameRequest):
    """
    Receives a single frame, runs YOLO inference, and returns detections.
    """
    yolo_model = ml_models.get("yolo")
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model is not available.")

    try:
        # Decode base64 string
        img_str = request.frame_data.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame from base64 string.")

        # Run YOLO inference (convert BGR->RGB for the model)
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            frame_rgb = frame
        inference_start_time = time.time()
        # Use reasonable defaults for confidence and image size; avoid half precision forcing
        try:
            results = yolo_model(frame_rgb, verbose=False, imgsz=640, conf=0.25)
        except TypeError:
            # Fallback if signature differs
            results = yolo_model.predict(frame_rgb, imgsz=640, conf=0.25)
        inference_time = (time.time() - inference_start_time) * 1000  # ms

        # Parse detections
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            img_height, img_width = frame.shape[:2]
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = yolo_model.names.get(class_id, "unknown").lower()
                
                if class_name in ["player", "ball", "referee", "person"]:
                    detections.append(Detection(
                        x=float(x1 / img_width),
                        y=float(y1 / img_height),
                        width=float((x2 - x1) / img_width),
                        height=float((y2 - y1) / img_height),
                        confidence=float(box.conf[0].cpu().numpy()),
                        class_name="player" if class_name == "person" else class_name
                    ))

        return FrameResponse(
            frame_id=request.frame_id,
            inference_time=inference_time,
            detections=detections
        )

    except Exception as e:
        logger.error(f"Error processing frame {request.frame_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
