#!/usr/bin/env python3
"""
🎬 Quick Video Prediction Example
Chelsea vs Swansea Match Analysis with YOLO Ball & Player Detection
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from video_prediction import VideoPredictor

def main():
    """Run prediction on Chelsea vs Swansea video"""
    
    print("🎯 YOLO Player & Ball Detection - Chelsea vs Swansea")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "/home/alperen/fomac/FoMAC/model-training/ball-detection/models/player_ball_detector/weights/best.pt"
    VIDEO_PATH = "/home/alperen/bitirme/soccerNet/england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea/1_720p.mkv"
    NUM_FRAMES = 100
    FRAMES_PER_GROUP = 10
    CONFIDENCE = 0.1
    OUTPUT_DIR = "predictions_output"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please make sure you have trained the model first!")
        return
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        print("Please check the video path!")
        return
    
    print(f"🏋️ Model: {MODEL_PATH}")
    print(f"🎬 Video: {VIDEO_PATH}")
    print(f"🎲 Total frames: {NUM_FRAMES}")
    print(f"📊 Frames per group: {FRAMES_PER_GROUP}")
    print(f"🎯 Confidence threshold: {CONFIDENCE}")
    print()
    
    try:
        # Create predictor
        predictor = VideoPredictor(MODEL_PATH, CONFIDENCE)
        
        # Process video
        results = predictor.process_video(
            video_path=VIDEO_PATH,
            num_frames=NUM_FRAMES,
            frames_per_group=FRAMES_PER_GROUP,
            output_dir=OUTPUT_DIR
        )
        
        print()
        print("🎉 BEFORE/AFTER Comparison completed successfully!")
        print("📊 Final Results:")
        print(f"   Groups processed: {results['total_groups']}")
        print(f"   Frames per group: {results['frames_per_group']}")
        print(f"   Total frames: {results['total_frames_processed']}")
        print()
        print("🔴 BEFORE Training (Player-only model):")
        print(f"   Players detected: {results['before_predictions']['Player']}")
        print(f"   Balls detected: {results['before_predictions']['Ball']}")
        print()
        print("🟢 AFTER Training (Player + Ball model):")
        print(f"   Players detected: {results['after_predictions']['Player']}")
        print(f"   Balls detected: {results['after_predictions']['Ball']}")
        print()
        print("📈 IMPROVEMENT:")
        print(f"   Players: +{results['improvement']['Player']}")
        print(f"   Balls: +{results['improvement']['Ball']}")
        print(f"   Processing time: {results['processing_time']:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()