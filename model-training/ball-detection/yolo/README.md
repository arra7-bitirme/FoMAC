# YOLO Multi-Class Detection Training Pipeline

A modular, configuration-driven training pipeline for YOLO-based multi-class detection (Player, Ball, Referee) using transfer learning.

## 🏗️ Architecture

The pipeline is structured with the following components for **3-class detection**:

### 📁 Directory Structure

```
yolo/
├── configs/                 # Configuration files
│   ├── paths.yaml          # Path configurations
│   ├── yolo_params.yaml    # YOLO model parameters (RTX 5070 optimized)
│   ├── extraction.yaml     # Dataset extraction settings
│   └── device.yaml         # Device selection settings
├── modules/                 # Core modules
│   ├── device_manager.py   # Device selection (CUDA/DirectML/CPU/Intel iGPU)
│   ├── dataset_extractor.py # Dataset preparation logic
│   └── trainer.py          # YOLO training with transfer learning
├── utils/                   # Utility functions
│   ├── config_utils.py     # Configuration and CLI management
│   └── visualization_utils.py # Visualization helpers
├── reports/                 # Report generation
│   └── report_generator.py # Excel/HTML training reports
├── main.py                 # Main orchestrator
├── visualize_labels.py     # Dataset quality visualization
├── visualize_specific_class.py # Class-specific inspection
└── requirements.txt        # Dependencies
```

### 🧩 Core Modules

1. **Device Manager**: Multi-platform device support
   - NVIDIA CUDA (preferred for RTX 5070)
   - Intel iGPU via Intel Extension for PyTorch
   - DirectML (Windows GPU fallback)
   - CPU with multi-threading

2. **Dataset Manager**: Handles ballDataset with 3 classes
   - Player (0): 86% of labels
   - Ball (1): 5.7% of labels (challenging small objects)
   - Referee (2): 8.4% of labels

3. **Trainer**: Transfer learning from player detection model
   - Loads pre-trained football_detector_optimized
   - Adapts to 3-class output (Player + Ball + Referee)
   - Advanced hyperparameter optimization

4. **Report Generator**: Performance analysis and visualization
   - Training metrics comparison
   - Class-specific performance breakdown
   - Excel and HTML outputs

## 🚀 Usage

### Quick Start

```bash
# Full training with transfer learning
python main.py --epochs 50

# Quick test with small dataset fraction  
python main.py --epochs 5 --fraction 0.01 --batch 4

# GPU training (RTX 5070 optimized)
python main.py --epochs 50 --batch 16 --device cuda:0

# Intel iGPU training (Linux)
python main.py --epochs 30 --batch 8 --device xpu
```

### Device-Specific Training

```bash
# RTX 5070 optimization
python main.py --epochs 50 --batch 16 --workers 12 --device cuda:0 --imgsz 1024

# Intel iGPU (requires intel-extension-for-pytorch)
python main.py --epochs 30 --batch 8 --device xpu --workers 4

# CPU optimization
python main.py --epochs 20 --batch 4 --workers 8 --device cpu

# DirectML (Windows GPU)
python main.py --epochs 30 --batch 8 --device dml
```

### Data Analysis

```bash
# Visualize dataset quality
python visualize_labels.py

# Inspect specific class (Ball detection focus)
python visualize_specific_class.py --class_id 1  # Ball class
python visualize_specific_class.py --class_id 2  # Referee class
```

## ⚙️ Configuration

### YOLO Parameters (`configs/yolo_params.yaml`) - RTX 5070 Optimized

```yaml
# Training settings
epochs: 50
batch: 16          # Optimized for RTX 5070
imgsz: 1024        # High resolution for small ball detection
workers: 12        # Multi-core utilization

# Performance optimizations
cache: true        # RAM caching for speed
amp: true         # Automatic Mixed Precision
multi_scale: true  # Multi-scale training

# Loss weights (class imbalance handling)
box: 7.5          # Bounding box loss
cls: 0.4          # Classification loss  
dfl: 2.0          # Distribution focal loss

# Ball detection specific
conf: 0.01        # Low confidence threshold for ball
iou: 0.65         # IoU threshold for NMS
max_det: 1000     # Max detections per image
```

### Device Configuration (`configs/device.yaml`)

```yaml
device_priority:
  - "cuda"     # NVIDIA RTX 5070 (preferred)
  - "xpu"      # Intel iGPU via IPEX
  - "dml"      # DirectML (Windows)
  - "cpu"      # CPU fallback

cuda:
  test_kernel: true
  fallback_on_error: true
  
intel_gpu:
  test_ops: true
  memory_fraction: 0.8

cpu:
  threads: 12    # Match your CPU cores
```

## 🎯 Transfer Learning Pipeline

### Model Flow
```
Player Detection Model (Pre-trained)
    ↓ (Load weights)
YOLO11n Architecture  
    ↓ (Adapt output layer)
3-Class Detection (Player + Ball + Referee)
    ↓ (Fine-tuning)
Ball Detection Specialist
```

### Performance Expectations

| Class | Baseline | Target | Challenge Level |
|-------|----------|--------|----------------|
| Player | 0.80+ | 0.85+ | Low (transfer learning) |
| Ball | 0.00 | 0.30+ | High (small objects) |
| Referee | 0.00 | 0.60+ | Medium (similar to players) |

## 🔧 Advanced Features

### Class Imbalance Handling
- **Weighted loss functions** for ball detection
- **Data augmentation** focused on small objects
- **High resolution training** (imgsz=1024) for ball visibility

### Hardware Optimization
- **RTX 5070 configuration** with optimal batch sizes
- **Intel iGPU support** via Intel Extension for PyTorch
- **Multi-threading optimization** for CPU training
- **Automatic Mixed Precision** for speed/memory balance

### Monitoring & Analysis
- **Real-time training metrics** via Ultralytics
- **Custom report generation** with class breakdowns
- **Dataset visualization tools** for quality control
- **Performance comparison** between baseline and fine-tuned models

## 📊 Dataset Information

### ballDataset Structure
```
ballDataset/
├── data.yaml              # YOLO dataset configuration
├── images/
│   ├── train/ (8,677)     # Training images
│   └── test/ (2,996)      # Test images  
└── labels/
    ├── train/             # YOLO format labels
    └── test/              # YOLO format labels
```

### Class Distribution
- **Total instances:** 14,188
- **Player (0):** 12,204 instances (86.0%)
- **Ball (1):** 808 instances (5.7%) - *Most challenging*
- **Referee (2):** 1,176 instances (8.3%)

## 🐛 Troubleshooting

### Common Issues

**Poor Ball Detection Performance**
```bash
# Increase image resolution for small objects
python main.py --imgsz 1280 --epochs 50

# Focus on ball class with lower confidence
python main.py --conf 0.005 --epochs 50
```

**GPU Memory Issues**
```bash
# Reduce batch size for your GPU
python main.py --batch 8 --epochs 30  # RTX 3060
python main.py --batch 4 --epochs 30  # RTX 2060
```

**Intel iGPU Not Working**
```bash
# Install Intel Extension
pip install intel-extension-for-pytorch

# Test availability
python -c "import intel_extension_for_pytorch as ipex; print('Intel GPU available:', ipex.xpu.is_available())"
```

**Slow CPU Training**
```bash
# Use data fraction for development
python main.py --fraction 0.001 --epochs 2 --batch 2
```

## 📋 Requirements

### Core Dependencies
```bash
pip install ultralytics>=8.0.0
pip install torch>=2.0.0 torchvision torchaudio
pip install opencv-python numpy pandas matplotlib
pip install pyyaml tqdm pillow
```

### Optional GPU Support
```bash
# Intel iGPU (Linux/Windows)
pip install intel-extension-for-pytorch

# DirectML (Windows)
pip install torch-directml

# NVIDIA CUDA (automatic with PyTorch CUDA)
```

## 🎯 Performance Targets

### Training Speed (RTX 5070)
- **1 epoch:** ~5-8 minutes (full dataset)
- **Inference:** ~10ms per image
- **Memory usage:** ~8GB VRAM (batch=16)

### Intel iGPU Performance (i7-12700H Iris Xe)
- **Expected speedup:** 2-4x vs CPU
- **1 epoch:** ~15-20 minutes (full dataset)
- **Recommended batch:** 8

### CPU Performance (i7-12700H)
- **1 epoch:** ~30-60 minutes (full dataset)
- **Recommended batch:** 4
- **Workers:** 8-12

## 🔄 Development Workflow

1. **Dataset verification**
   ```bash
   python visualize_labels.py
   ```

2. **Quick functionality test**
   ```bash
   python main.py --fraction 0.001 --epochs 2
   ```

3. **Hardware-specific optimization**
   ```bash
   python main.py --epochs 5 --batch [optimal_batch]
   ```

4. **Full training**
   ```bash
   python main.py --epochs 50
   ```

5. **Performance analysis**
   ```bash
   # Check reports folder
   ls reports/player_ball_detector/
   ```

This multi-class detection pipeline leverages transfer learning from player detection to achieve ball and referee detection capabilities.

## 🏗️ Architecture

The pipeline is structured as microservices with the following components:

### 📁 Directory Structure

```
yolo/
├── configs/                 # Configuration files
│   ├── paths.yaml          # Path configurations
│   ├── yolo_params.yaml    # YOLO model parameters
│   ├── extraction.yaml     # Dataset extraction settings
│   └── device.yaml         # Device selection settings
├── modules/                 # Core modules
│   ├── device_manager.py   # Device selection and management
│   ├── dataset_extractor.py # Dataset extraction logic
│   └── trainer.py          # YOLO training logic
├── utils/                   # Utility functions
│   ├── config_utils.py     # Configuration management
│   └── visualization_utils.py # Visualization helpers
├── main.py                 # Main orchestrator
└── __init__.py            # Package initialization
```

### 🧩 Modules

1. **Device Manager**: Handles CUDA/DirectML/CPU device selection with automatic fallback
2. **Dataset Extractor**: Extracts frames and labels from SoccerNet with alignment preservation
3. **Trainer**: YOLO model training with configuration-driven parameters
4. **Config Manager**: Loads YAML configs and merges with CLI arguments
5. **Visualization Utils**: Helper functions for bbox scaling and frame processing

## 🚀 Usage

### Basic Usage

```bash
# Run with default configurations
python main.py

# Run with custom parameters
python main.py --epochs 200 --batch 32 --lr0 0.001
```

### Configuration-Only Mode

```bash
# Edit configuration files, then run
python main.py
```

### CLI Override Mode

```bash
# Override specific parameters via command line
python main.py --run-extraction --max-samples-per-half 1000 --epochs 150
```

### Specialized Operations

```bash
# Extract dataset only
python main.py --extract-only

# Train only (skip extraction)
python main.py --train-only

# Evaluate trained model
python main.py --evaluate

# Run inference
python main.py --predict /path/to/images_or_video
```

## ⚙️ Configuration

### Path Configuration (`configs/paths.yaml`)

```yaml
workspace_root: "~/fomac/FoMAC/model-training/ball-detection"
soccernet_root: "soccerNet"  # Not used for ball detection
output_root: "datasets/ball_detection_extracted"
models_root: "models"
```

### YOLO Parameters (`configs/yolo_params.yaml`)

```yaml
model: "yolo11n.pt"  # Pretrained model for transfer learning
epochs: 50
batch: 16
imgsz: 640
lr0: 0.002
project_name: "ball_detector"
# ... and many more parameters
```

### Extraction Settings (`configs/extraction.yaml`)

```yaml
run_extraction: true
max_samples_per_half: null  # null for all samples
class_names:
  - "ball"  # Primary target class for ball detection
```

### Device Configuration (`configs/device.yaml`)

```yaml
device_priority:
  - "cuda"     # NVIDIA CUDA (preferred)
  - "dml"      # DirectML (Windows GPU fallback)
  - "cpu"      # CPU (final fallback)
```

## 🔧 Features

### Configuration Management
- **YAML-based configuration files** for all parameters
- **CLI argument support** with automatic override
- **Validation** of configuration parameters
- **Flexible path handling** with workspace-relative paths

### Device Management
- **Automatic device selection** with fallback chain
- **CUDA kernel testing** to ensure compatibility
- **DirectML support** for Windows GPU acceleration
- **CPU fallback** with thread configuration

### Dataset Extraction
- **Hugging Face dataset support** for modern dataset formats
- **Automatic YOLO format conversion** from arrow/parquet files
- **Flexible dataset structure detection** with multiple fallbacks
- **Ball detection focus** with single-class optimization

### Training Pipeline
- **Transfer learning** with pretrained YOLO11n model
- **Configuration-driven training** with full YOLO parameter support
- **Automatic mixed precision** based on device capabilities
- **Progress logging** with detailed status updates
- **Model evaluation and inference** support

### Logging & Monitoring
- **Structured logging** with configurable levels
- **File logging** support
- **Progress tracking** with detailed status updates
- **Error handling** with graceful fallbacks

## 🔄 Migration from Player Detection

The original player detection code has been adapted for ball detection:

1. **Model configuration** changed to use pretrained `yolo11n.pt` for transfer learning
2. **Dataset extractor** rewritten to handle Hugging Face dataset format
3. **Class configuration** focused on ball detection (single class)
4. **Project naming** updated to `ball_detector`
5. **Path configuration** updated for ball detection workspace

### Key Improvements

- ✅ **Transfer learning support** with pretrained YOLO models
- ✅ **Hugging Face dataset integration** for modern ML workflows
- ✅ **Simplified configuration** focused on ball detection
- ✅ **Automatic format conversion** from various dataset structures
- ✅ **Preserved modularity** with ball-specific optimizations
- ✅ **Ready for training** with proper YOLO format output

## 📋 Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- PyYAML
- tqdm
- Hugging Face datasets
- Pillow

Optional:
- torch-directml (for Windows GPU acceleration)

## 🏃‍♂️ Quick Start

1. **Configure paths** in `configs/paths.yaml`
2. **Adjust training parameters** in `configs/yolo_params.yaml` if needed
3. **Run the pipeline**:
   ```bash
   cd yolo
   python main.py
   ```

## 🎯 Example Workflows

### Development Workflow
```bash
# Install dependencies
pip install -r requirements.txt

# Extract ball dataset for testing
python main.py --extract-only

# Quick training test
python main.py --train-only --epochs 10 --batch 8

# Full training with transfer learning
python main.py --epochs 50
```

### Production Workflow
```bash
# Full extraction and training
python main.py

# Evaluation
python main.py --evaluate

# Inference on new data
python main.py --predict /path/to/new/videos
```

This ball detection pipeline uses transfer learning with pretrained YOLO models and supports modern dataset formats for efficient training.