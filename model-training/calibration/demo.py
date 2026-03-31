import os
import sys
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image, ImageDraw
import requests
import argparse

# Add the current directory to sys.path so we can import the local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib.utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool, 
                                            get_keypoints_from_heatmap_batch_maxpool_l, 
                                            complete_keypoints, 
                                            coords_to_dict)
from nbjw_calib.utils.utils_calib import FramebyFrameCalib
from sn_calibration_baseline.camera import Camera
from sn_calibration_baseline.soccerpitch import SoccerPitch
from ultralytics import YOLO
from team_clasifier import AutoLabEmbedder, AutomaticTeamClusterer
try:
    from boxmot import ByteTrack
except ImportError:
    print("❌ UYARI: 'boxmot' kütüphanesi eksik!")

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration for HRNet models (MATCHING nbjw_calib.yaml)
    # Keypoints Model Config
    cfg = {
        'MODEL': {
            'IMAGE_SIZE': [960, 540],
            'NUM_JOINTS': 58,
            'PRETRAIN': '',
            'EXTRA': {
                'FINAL_CONV_KERNEL': 1,
                'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_CHANNELS': [64], 'FUSE_METHOD': 'SUM'},
                'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [48, 96], 'FUSE_METHOD': 'SUM'},
                'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [48, 96, 192], 'FUSE_METHOD': 'SUM'},
                'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [48, 96, 192, 384], 'FUSE_METHOD': 'SUM'},
            }
        }
    }
    
    # Lines Model Config
    cfg_l = {
        'MODEL': {
            'IMAGE_SIZE': [960, 540],
            'NUM_JOINTS': 24,
            'PRETRAIN': '',
            'EXTRA': {
                'FINAL_CONV_KERNEL': 1,
                'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_CHANNELS': [64], 'FUSE_METHOD': 'SUM'},
                'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [48, 96], 'FUSE_METHOD': 'SUM'},
                'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [48, 96, 192], 'FUSE_METHOD': 'SUM'},
                'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [48, 96, 192, 384], 'FUSE_METHOD': 'SUM'},
            }
        }
    }

    # Model paths
    checkpoint_kp = "SV_kp.pth"
    checkpoint_l = "SV_lines.pth"

    # Download models if not present
    # Using the zenodo links from nbjw_calib.py
    download_file("https://zenodo.org/records/12626395/files/SV_kp?download=1", checkpoint_kp)
    download_file("https://zenodo.org/records/12626395/files/SV_lines?download=1", checkpoint_l)

    # Load models
    model_kp = get_cls_net(cfg)
    try:
        model_kp.load_state_dict(torch.load(checkpoint_kp, map_location=device))
    except RuntimeError:
         # Sometimes models are saved with 'module.' prefix if trained on multi-gpu
        state_dict = torch.load(checkpoint_kp, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_kp.load_state_dict(new_state_dict)
        
    model_kp.to(device)
    model_kp.eval()

    model_l = get_cls_net_l(cfg_l)
    try:
        model_l.load_state_dict(torch.load(checkpoint_l, map_location=device))
    except RuntimeError:
        state_dict = torch.load(checkpoint_l, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_l.load_state_dict(new_state_dict)
        
    model_l.to(device)
    model_l.eval()

    # Load YOLO model for player/ball detection
    print("Loading YOLO model...")
    model_yolo = YOLO("best.pt") # Using yolo11m usually, or yolov8m

    embedder = AutoLabEmbedder()
    clusterer = AutomaticTeamClusterer()

    # Transformations
    # NBJW expects 960x540 input
    tfms_resize = T.Compose([
        T.Resize((540, 960)),
        T.ToTensor()
    ])

    # ---------------------------------------------------------
    # PART 1: BATCH PROCESSING
    # ---------------------------------------------------------
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data_video', help='path to input directory or video file')
    args = parser.parse_args()

    output_dir = "output_video"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_writer = None
    source_basename = os.path.basename(os.path.normpath(args.source))
    video_output_name = f"{os.path.splitext(source_basename)[0]}_output.mp4"
    video_output_path = os.path.join(output_dir, video_output_name)
    
    is_video = False
    cap = None
    if os.path.isfile(args.source) and args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        is_video = True
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Found video {args.source} with {total_frames} frames.")
    else:
        input_dir = args.source
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            print(f"Created {input_dir}. Please put images there.")
            return

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"No images found in {input_dir}")
            return

        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
        print(f"Found {len(image_files)} images to process.")
        fps = 25

    try:
        tracker = ByteTrack(reid_weights=None, device='cuda:0' if device.type=='cuda' else 'cpu', half=True, frame_rate=int(fps))
    except NameError:
        tracker = None

    team_passes = {0: 0, 1: 0}
    team_streaks = {0: 0, 1: 0}
    last_pass_info = {'distance': 0.0, 'team_id': -1}
    last_possessor = None # {'track_id': id, 'team_id': id, 'pos': array}
    potential_possessor = None
    potential_frames = 0
    frames_since_loss = 0

    frame_idx = 0
    while True:
        if is_video:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            original_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            image_file = f"frame_{frame_idx:06d}.jpg"
            print(f"\nProcessing {image_file}...")
        else:
            if frame_idx >= len(image_files):
                break
            image_file = image_files[frame_idx]
            print(f"\nProcessing {image_file}... ({frame_idx+1}/{len(image_files)})")
            image_path = os.path.join(input_dir, image_file)
            try:
                original_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Could not open {image_file}: {e}")
                frame_idx += 1
                continue

        original_width, original_height = original_image.size
        
        # ---------------------------------------------------------
        # DETECT PLAYERS AND BALL
        # ---------------------------------------------------------
        # print("Detecting objects...")
        results = model_yolo(original_image, verbose=False)
        all_dets = results[0].boxes.data.cpu().numpy()
        
        person_dets = []
        ball_boxes = []
        
        for det in all_dets:
            if det[4] < 0.3: continue
            cls_id = int(det[5])
            if cls_id == 0:
                person_dets.append(det)
            elif cls_id == 1:
                ball_boxes.append(det[:4])
                
        image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        # Tracking setup
        person_data = []
        if len(person_dets) > 0 and tracker is not None:
            tracks = tracker.update(np.array(person_dets), image_cv)
            for track in tracks:
                bbox = track[:4].astype(int)
                track_id = int(track[4])
                
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_cv.shape[1], x2), min(image_cv.shape[0], y2)
                
                team_id = -1
                if y2 - y1 > 5 and x2 - x1 > 5:
                    crop = image_cv[y1:y2, x1:x2]
                    feat = embedder.get_features(crop)
                    if feat is not None:
                        if not clusterer.trained:
                            clusterer.collect(feat)
                            if len(clusterer.data_bank) >= 50:
                                clusterer.train()
                        if clusterer.trained:
                            team_id = clusterer.predict(feat)
                
                person_data.append((bbox, team_id, track_id))
        else:
            # Fallback if no tracker
            for det in person_dets:
                bbox = det[:4].astype(int)
                person_data.append((bbox, -1, -1))
                
        person_boxes = person_data

        # Process image for calibration
        # Note: NBJW was trained on 960x540
        img_tensor = tfms_resize(original_image).unsqueeze(0).to(device)

        # print("Running inference...")
        with torch.no_grad():
            heatmaps = model_kp(img_tensor)
            heatmaps_l = model_l(img_tensor)

        # print("Extracting keypoints...")
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        
        # Use 0.05 threshold to capture more points
        kp_dict = coords_to_dict(kp_coords, threshold=0.05)
        lines_dict = coords_to_dict(line_coords, threshold=0.05)

        # Note: complete_keypoints expects width/height of the resized image (960x540)
        # The output keypoints are normalized (0-1) range if normalize=True
        final_dict = complete_keypoints(kp_dict, lines_dict, w=960, h=540, normalize=True)
        
        # Only taking the first image in batch
        # final_dict is a list of dicts
        keypoints_prediction = final_dict[0]
        
        print(f"Detected {len(keypoints_prediction)} keypoints.")

        # ---------------------------------------------------------
        # PART 2: CALIBRATION - Compute Parameters
        # ---------------------------------------------------------
        
        # print("Computing calibration...")
        # FramebyFrameCalib expects original image dimensions to denormalize correctly
        cam = FramebyFrameCalib(iwidth=original_width, iheight=original_height, denormalize=True)
        
        # Update camera with predictions
        cam.update(keypoints_prediction)
        
        # Heuristic voting to find best calibration
        # It tries different RANSAC thresholds and modes (full, ground_plane, main)
        # Returns the best result based on reprojection error
        calibration_result = cam.heuristic_voting()
        
        import math
        if calibration_result and math.isnan(calibration_result.get('rep_err', 0.0)):
            calibration_result = False
            
        if calibration_result:
            print("Calibration Successful!")
            # print(f"Mode used: {calibration_result['mode']}")
            print(f"Reprojection Error: {calibration_result['rep_err']:.4f}")
            
            params = calibration_result['cam_params']
            # print("Camera Parameters:")
            # print(f"  Pan: {params['pan_degrees']:.2f}")
            # print(f"  Tilt: {params['tilt_degrees']:.2f}")
            # print(f"  Roll: {params['roll_degrees']:.2f}")
            # print(f"  Focal Length (x,y): {params['x_focal_length']:.2f}, {params['y_focal_length']:.2f}")
            # print(f"  Position (x,y,z): {params['position_meters']}")
            
            # Homography
            # You can compute homography from these parameters if needed, or use the ground plane method
            # print("Computing Homography from 3D projection...")
            
            # ---------------------------------------------------------
            # PART 3: VISUALIZATION
            # ---------------------------------------------------------
            # print("Creating visualization...")
            
            # Initialize Camera model
            camera = Camera(iwidth=original_width, iheight=original_height)
            camera.from_json_parameters(params)
            
            # Convert PIL image to OpenCV BGR format (moved up for team crop)

            
            # Draw pitch lines
            camera.draw_pitch(image_cv)
            
            # Save output
            # ---------------------------------------------------------
            # PART 4: POSSESSION & PASS LOGIC
            # ---------------------------------------------------------
            # Calculate 3D/2D positions for ball and players
            ball_pos = None
            if len(ball_boxes) > 0:
                bbox = ball_boxes[0]
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2 + (bbox[3] - bbox[1]) * 0.4
                try:
                    ball_pos = camera.unproject_point_on_planeZ0(np.array([center_x, center_y]))
                except:
                    ball_pos = None
            
            # Evaluate posession
            current_possessor = None
            if ball_pos is not None:
                min_dist = float('inf')
                for item in person_boxes:
                    bbox = item[0]
                    team_id = item[1]
                    track_id = item[2] if len(item) > 2 else -1
                    
                    foot_x = (bbox[0] + bbox[2]) / 2
                    foot_y = bbox[3]
                    try:
                        p_pos = camera.unproject_point_on_planeZ0(np.array([foot_x, foot_y]))
                        dist = np.linalg.norm(p_pos - ball_pos)
                        if dist < 2.5 and dist < min_dist: # 2.5 meters radius
                            min_dist = dist
                            current_possessor = {'track_id': track_id, 'team_id': team_id, 'pos': p_pos}
                    except:
                        continue
            
            if current_possessor is not None and current_possessor['team_id'] != -1 and current_possessor['track_id'] != -1:
                # We have a valid player controlling the ball
                frames_since_loss = 0
                if last_possessor is None:
                    last_possessor = current_possessor
                else:
                    is_same_team = (last_possessor['team_id'] == current_possessor['team_id'])
                    is_different_id = (last_possessor['track_id'] != current_possessor['track_id'])
                    
                    dist_between_players = np.linalg.norm(last_possessor['pos'] - current_possessor['pos'])
                    is_physically_different = (dist_between_players > 3.0)

                    if is_same_team:
                        if is_different_id and is_physically_different:
                            # Potential pass within the same team
                            if potential_possessor is not None and potential_possessor['track_id'] == current_possessor['track_id']:
                                potential_frames += 1
                            else:
                                potential_possessor = current_possessor
                                potential_frames = 1
                                
                            # If held for 6 frames -> confirmed pass!
                            if potential_frames >= 6:
                                team_passes[current_possessor['team_id']] += 1
                                pass_dist = np.linalg.norm(last_possessor['pos'] - current_possessor['pos'])
                                team_streaks[current_possessor['team_id']] += 1
                                team_streaks[1 - current_possessor['team_id']] = 0 # reset other team's streak
                                last_pass_info = {'distance': pass_dist, 'team_id': current_possessor['team_id']}
                                
                                print(f"PASS DETECTED! Team {current_possessor['team_id']} (Dist: {pass_dist:.1f}m, Streak: {team_streaks[current_possessor['team_id']]})")
                                last_possessor = current_possessor
                                potential_possessor = None
                                potential_frames = 0
                        else:
                            # Same player or ID switch but same physical location
                            last_possessor['pos'] = current_possessor['pos'] # update position
                            potential_possessor = None
                            potential_frames = 0
                    else:
                        # Different team = Turnover / Interception
                        # No physical distance check required for turnover, just hold for 3 frames
                        if potential_possessor is not None and potential_possessor['track_id'] == current_possessor['track_id']:
                            potential_frames += 1
                        else:
                            potential_possessor = current_possessor
                            potential_frames = 1
                            
                        if potential_frames >= 5:
                            team_streaks[0] = 0
                            team_streaks[1] = 0
                            last_possessor = current_possessor
                            potential_possessor = None
                            potential_frames = 0
            else:
                 frames_since_loss += 1
                 if frames_since_loss > 15:
                     potential_possessor = None
                     potential_frames = 0

            # Draw Scoreboard on image_cv
            cv2.rectangle(image_cv, (10, 10), (450, 80), (0, 0, 0), -1)
            cv2.putText(image_cv, "SCOREBOARD", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_cv, f"Mavi T(0): {team_passes[0]} Pass", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            cv2.putText(image_cv, f"Kirmizi T(1): {team_passes[1]} Pass", (230, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

            # Draw Pass Status on Top Right
            tr_x = original_width - 320
            cv2.rectangle(image_cv, (tr_x, 10), (original_width - 10, 80), (0, 0, 0), -1)
            cv2.putText(image_cv, "LAST PASS INFO", (tr_x + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            p_dist = last_pass_info['distance']
            p_team = last_pass_info['team_id']
            if p_team != -1:
                t_color = (255, 100, 100) if p_team == 0 else (100, 100, 255)
                cv2.putText(image_cv, f"Mesafe: {p_dist:.1f}m | Seri: {team_streaks[p_team]}", (tr_x + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_color, 2)
            else:
                cv2.putText(image_cv, "Bekleniyor...", (tr_x + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # ---------------------------------------------------------
            # PART 5: MINIMAP OVERLAY
            # ---------------------------------------------------------
            print("Creating minimap...")
            minimap = draw_minimap(camera, person_boxes, ball_boxes)
            
            # Resize minimap to fit nicely (e.g., 25% of image width)
            target_width = int(original_width * 0.25)
            scale_ratio = target_width / minimap.shape[1]
            target_height = int(minimap.shape[0] * scale_ratio)
            
            minimap_resized = cv2.resize(minimap, (target_width, target_height))
            
            # Position: Bottom Center
            x_offset = (original_width - target_width) // 2
            y_offset = original_height - target_height - 20 # 20px padding from bottom
            
            # Overlay
            # Create ROI
            roi = image_cv[y_offset:y_offset+target_height, x_offset:x_offset+target_width]
            
            # Blend (Optional: make it slightly transparent or just opaque)
            # For now, opaque with a border
            image_cv[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = minimap_resized
            
            # Add border
            cv2.rectangle(image_cv, (x_offset, y_offset), (x_offset+target_width, y_offset+target_height), (255, 255, 255), 2)
            
            # Save final output
            # Overwrite the previous _vis.jpg or create a new one? 
            # User output expectation: "minimapi outputta fotografin orta altina koy"
            # I will overwrite the _vis.jpg effectively or just save it as the final result.
            
            # Let's save it as the standard output
            # output_filename = f"{os.path.splitext(image_file)[0]}_vis.jpg"
            # output_path = os.path.join(output_dir, output_filename)
            # cv2.imwrite(output_path, image_cv)
            # print(f"Visualization with minimap saved to {output_path}")

        else:
            print("Calibration failed. Not enough keypoints or unstable solution.")

        # Write to Video (always write to maintain fps and sync)
        if video_writer is None:
            height, width, _ = image_cv.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        video_writer.write(image_cv)
            
        frame_idx += 1

    if is_video and cap is not None:
        cap.release()

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {video_output_path}")

def draw_minimap(camera, persons=[], balls=[]):
    # Canvas settings
    scale = 8 # pixels per meter
    margin = 50
    pitch_length = 105
    pitch_width = 68
    
    img_width = int(pitch_length * scale + 2 * margin)
    img_height = int(pitch_width * scale + 2 * margin)
    
    # Green background
    minimap = np.ones((img_height, img_width, 3), dtype=np.uint8) * 50
    minimap[:, :, 1] = 100 # Dark green

    # World to Image transform
    def world_to_minimap(pt3d):
        # pt3d is (x, y, z)
        # origin is center of pitch
        # x is length, y is width
        # map x: [-L/2, L/2] -> [margin, width-margin]
        # map y: [-W/2, W/2] -> [margin, height-margin]
        
        mx = int((pt3d[0] + pitch_length/2) * scale + margin)
        my = int((pt3d[1] + pitch_width/2) * scale + margin)
        return (mx, my)

    # Draw Pitch Lines
    field = SoccerPitch()
    polylines = field.sample_field_points()
    for name, line in polylines.items():
        pts = [world_to_minimap(p) for p in line]
        # Draw polylines
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(minimap, [pts], False, (255, 255, 255), 2)

    # Draw Camera Position
    cam_pos = camera.position
    cx, cy = world_to_minimap(cam_pos)
    
    # Draw FOV
    # Project image corners to ground
    h, w = camera.image_height, camera.image_width
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    fov_pts = []
    
    # Only draw FOV if camera is above ground roughly
    if cam_pos[2] < 0: 
        pass

    for px, py in corners:
        try:
            # unproject_point_on_planeZ0 might fail for horizon points
            ground_pt = camera.unproject_point_on_planeZ0(np.array([px, py]))
            
            # Check if point is reasonably close (e.g. within 200m) to avoid infinite lines
            if np.linalg.norm(ground_pt) < 150:
                fov_pts.append(world_to_minimap(ground_pt))
        except:
            pass
            
    if len(fov_pts) == 4:
         pts = np.array(fov_pts, dtype=np.int32).reshape((-1, 1, 2))
         cv2.polylines(minimap, [pts], True, (0, 255, 255), 2) # Yellow FOV
         
         # Draw detailed FOV lines from camera
         for pt in fov_pts:
             cv2.line(minimap, (cx, cy), pt, (0, 255, 255), 1)
             
    # Draw Camera Dot on top
    cv2.circle(minimap, (cx, cy), 8, (0, 0, 255), -1) # Red dot for camera

    # ---------------------------------------------------------
    # DRAW OBJECTS (PLAYERS & BALL)
    # ---------------------------------------------------------
    
    # Project Persons
    for item in persons:
        track_id = -1
        if isinstance(item, tuple):
            if len(item) >= 2:
                bbox = item[0]
                team_id = item[1]
            if len(item) >= 3:
                track_id = item[2]
        else:
            bbox = item
            team_id = -1

        # bbox is [x1, y1, x2, y2]
        # Use bottom center point for projection
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]
        
        try:
            ground_pt = camera.unproject_point_on_planeZ0(np.array([foot_x, foot_y]))
            if np.linalg.norm(ground_pt) < 100: # Sanity check inside pitch area roughly
                 mx, my = world_to_minimap(ground_pt)
                 
                 if team_id == 0:
                     dot_c = (255, 50, 50)   # Team 0: Blue
                 elif team_id == 1:
                     dot_c = (50, 50, 255)   # Team 1: Red
                 else:
                     dot_c = (200, 200, 200) # Gray/Unknown
                     
                 # Draw player dot
                 # Blue with white outline
                 cv2.circle(minimap, (mx, my), 6, (255, 255, 255), -1) 
                 cv2.circle(minimap, (mx, my), 4, dot_c, -1)
        except Exception as e:
            pass

    # Project Balls
    for bbox in balls:
        # Use center point for ball
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2 + (bbox[3] - bbox[1]) * 0.4 # Slightly lower than center
        
        try:
            ground_pt = camera.unproject_point_on_planeZ0(np.array([center_x, center_y]))
            if np.linalg.norm(ground_pt) < 100:
                 mx, my = world_to_minimap(ground_pt)
                 # Draw ball dot (Yellow/Orange)
                 cv2.circle(minimap, (mx, my), 6, (0, 0, 0), -1) # outline
                 cv2.circle(minimap, (mx, my), 4, (0, 165, 255), -1)
        except:
            pass

    # cv2.imwrite(output_path, minimap)
    return minimap

if __name__ == "__main__":
    main()
