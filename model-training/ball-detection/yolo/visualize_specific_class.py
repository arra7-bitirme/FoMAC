#!/usr/bin/env python3
"""
Specific class visualization tool - focuses on images containing specific classes
"""

import cv2
import numpy as np
import os
import random
import argparse
from pathlib import Path
import yaml

def load_class_names(data_yaml_path):
    """Load class names from data.yaml file"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def read_yolo_labels(label_path):
    """Read YOLO format labels from file"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append((class_id, x_center, y_center, width, height))
    return labels

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to bounding box coordinates"""
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    
    return x1, y1, x2, y2

def draw_labels_on_image(image_path, label_path, class_names):
    """Draw YOLO labels on image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Load labels
    labels = read_yolo_labels(label_path)
    
    # Color map for different classes
    colors = {
        0: (0, 255, 0),    # Player - Green
        1: (0, 0, 255),    # Ball - Red  
        2: (255, 0, 0),    # Referee - Blue
    }
    
    # Draw bounding boxes
    for class_id, x_center, y_center, width, height in labels:
        x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
        
        color = colors.get(class_id, (255, 255, 255))
        class_name = class_names.get(class_id, f'Class_{class_id}')
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f'{class_name}'
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def find_images_with_class(dataset_path, target_class_id, split='train', max_images=20):
    """Find images containing specific class"""
    labels_dir = os.path.join(dataset_path, 'labels', split)
    images_dir = os.path.join(dataset_path, 'images', split)
    
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return []
    
    matching_images = []
    label_files = list(Path(labels_dir).glob('*.txt'))
    
    for label_file in label_files:
        labels = read_yolo_labels(str(label_file))
        
        # Check if target class exists in this image
        has_target_class = any(class_id == target_class_id for class_id, _, _, _, _ in labels)
        
        if has_target_class:
            # Check if corresponding image exists
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = os.path.join(images_dir, label_file.stem + ext)
                if os.path.exists(image_path):
                    matching_images.append((image_path, str(label_file)))
                    break
        
        if len(matching_images) >= max_images:
            break
    
    return matching_images

def visualize_class_samples(dataset_path, target_class_name, num_samples=10, split='train'):
    """Visualize samples containing specific class"""
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    class_names = load_class_names(data_yaml_path)
    
    # Find class ID
    target_class_id = None
    for class_id, class_name in class_names.items():
        if class_name.lower() == target_class_name.lower():
            target_class_id = class_id
            break
    
    if target_class_id is None:
        print(f"Class '{target_class_name}' not found. Available classes: {list(class_names.values())}")
        return
    
    print(f"Looking for images containing '{target_class_name}' (class_id: {target_class_id})")
    
    # Find images with target class
    matching_images = find_images_with_class(dataset_path, target_class_id, split, num_samples * 2)
    
    if not matching_images:
        print(f"No images found containing class '{target_class_name}'")
        return
    
    # Randomly sample
    random.shuffle(matching_images)
    selected_images = matching_images[:num_samples]
    
    print(f"Found {len(matching_images)} images with '{target_class_name}', showing {len(selected_images)} samples")
    print(f"Class colors: Player=Green, Ball=Red, Referee=Blue")
    print("\nPress any key to show next image, 'q' to quit")
    
    for i, (image_path, label_path) in enumerate(selected_images):
        print(f"\n[{i+1}/{len(selected_images)}] Processing: {os.path.basename(image_path)}")
        
        # Draw labels on image
        img_with_labels = draw_labels_on_image(image_path, label_path, class_names)
        
        if img_with_labels is not None:
            # Count labels by class
            labels = read_yolo_labels(label_path)
            class_counts = {}
            for class_id, _, _, _, _ in labels:
                class_name = class_names.get(class_id, f'Class_{class_id}')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"Labels found: {class_counts}")
            
            # Resize image if too large
            height, width = img_with_labels.shape[:2]
            if height > 800 or width > 1200:
                scale = min(800/height, 1200/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_with_labels = cv2.resize(img_with_labels, (new_width, new_height))
            
            # Show image
            cv2.imshow(f'{target_class_name} Detection - {os.path.basename(image_path)}', img_with_labels)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
                
            cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Visualize specific class in YOLO dataset')
    parser.add_argument('--dataset', '-d', 
                       default='/home/alperen/fomac/FoMAC/model-training/ball-detection/ballDataset',
                       help='Path to dataset directory')
    parser.add_argument('--class', '-c', dest='target_class', required=True,
                       choices=['Player', 'Ball', 'Referee'],
                       help='Target class to visualize')
    parser.add_argument('--samples', '-n', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--split', '-s', choices=['train', 'test'], default='train',
                       help='Dataset split to visualize')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Dataset directory not found: {args.dataset}")
        return
    
    visualize_class_samples(args.dataset, args.target_class, args.samples, args.split)

if __name__ == "__main__":
    main()