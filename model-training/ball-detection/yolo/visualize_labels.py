#!/usr/bin/env python3
"""
Dataset Label Visualization Tool
Visualizes YOLO format labels on images to verify correct annotation
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

def visualize_random_samples(dataset_path, num_samples=10, split='train'):
    """Visualize random samples from dataset"""
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    class_names = load_class_names(data_yaml_path)
    
    images_dir = os.path.join(dataset_path, 'images', split)
    labels_dir = os.path.join(dataset_path, 'labels', split)
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(images_dir).glob(ext))
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    # Select random samples
    random.shuffle(image_files)
    selected_files = image_files[:num_samples]
    
    print(f"Visualizing {len(selected_files)} random samples from {split} set...")
    print(f"Class names: {class_names}")
    print("\nPress any key to show next image, 'q' to quit")
    
    for i, image_path in enumerate(selected_files):
        # Corresponding label file
        label_path = os.path.join(labels_dir, image_path.stem + '.txt')
        
        print(f"\n[{i+1}/{len(selected_files)}] Processing: {image_path.name}")
        
        # Draw labels on image
        img_with_labels = draw_labels_on_image(str(image_path), label_path, class_names)
        
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
            cv2.imshow(f'Dataset Visualization - {image_path.name}', img_with_labels)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
                
            cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()

def analyze_dataset_statistics(dataset_path):
    """Analyze dataset statistics"""
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    class_names = load_class_names(data_yaml_path)
    
    stats = {}
    
    for split in ['train', 'test']:
        labels_dir = os.path.join(dataset_path, 'labels', split)
        if not os.path.exists(labels_dir):
            continue
            
        split_stats = {
            'total_images': 0,
            'total_objects': 0,
            'class_counts': {name: 0 for name in class_names.values()},
            'images_per_class': {name: 0 for name in class_names.values()}
        }
        
        label_files = list(Path(labels_dir).glob('*.txt'))
        split_stats['total_images'] = len(label_files)
        
        for label_file in label_files:
            labels = read_yolo_labels(str(label_file))
            split_stats['total_objects'] += len(labels)
            
            classes_in_image = set()
            for class_id, _, _, _, _ in labels:
                class_name = class_names.get(class_id, f'Class_{class_id}')
                split_stats['class_counts'][class_name] += 1
                classes_in_image.add(class_name)
            
            for class_name in classes_in_image:
                split_stats['images_per_class'][class_name] += 1
        
        stats[split] = split_stats
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} SET:")
        print(f"  Total Images: {split_stats['total_images']}")
        print(f"  Total Objects: {split_stats['total_objects']}")
        print(f"  Avg Objects per Image: {split_stats['total_objects']/max(1, split_stats['total_images']):.2f}")
        
        print(f"\n  Object Counts by Class:")
        for class_name, count in split_stats['class_counts'].items():
            percentage = (count / max(1, split_stats['total_objects'])) * 100
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n  Images Containing Each Class:")
        for class_name, count in split_stats['images_per_class'].items():
            percentage = (count / max(1, split_stats['total_images'])) * 100
            print(f"    {class_name}: {count} images ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO dataset labels')
    parser.add_argument('--dataset', '-d', 
                       default='/home/alperen/fomac/FoMAC/model-training/ball-detection/ballDataset',
                       help='Path to dataset directory')
    parser.add_argument('--samples', '-n', type=int, default=10,
                       help='Number of random samples to visualize')
    parser.add_argument('--split', '-s', choices=['train', 'test'], default='train',
                       help='Dataset split to visualize')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics, no visualization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Dataset directory not found: {args.dataset}")
        return
    
    # Show statistics
    analyze_dataset_statistics(args.dataset)
    
    # Show visualizations
    if not args.stats_only:
        print(f"\nStarting visualization of {args.samples} samples from {args.split} set...")
        visualize_random_samples(args.dataset, args.samples, args.split)

if __name__ == "__main__":
    main()