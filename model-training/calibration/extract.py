import cv2
import os
import glob

print("Cleaning up old frames...")
for f in glob.glob('data_video/*.jpg'):
    os.remove(f)

video_path = "data/Leipzig 1-2 Beşiktaş _ UEFA Şampiyonlar Ligi Maç Özeti.mp4"
output_dir = "data_video"
os.makedirs(output_dir, exist_ok=True)

print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_frame = int(total_frames * 0.75)  # Jump 75% in

print(f"Total frames: {total_frames}. Automatically jumping to frame {start_frame} (75% mark)...")
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

count = 0

print("Extracting 100 frames from the middle...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or count >= 100: 
        break
    cv2.imwrite(os.path.join(output_dir, f"{count:06d}.jpg"), frame)
    count += 1
    
cap.release()
print(f"Done! Extracted {count} frames starting from {start_frame}.")
