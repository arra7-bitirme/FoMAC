# YOLO Football Player Detection Training Pipeline

A modular, configuration-driven training pipeline for YOLO-based football player detection using the SoccerNet dataset.

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
workspace_root: "~/bitirme"
soccernet_root: "soccerNet"
output_root: "datasets/yolo_detection_extracted"
models_root: "models"
```

### YOLO Parameters (`configs/yolo_params.yaml`)

```yaml
model: "yolo11n.yaml"
epochs: 120
batch: 16
imgsz: 640
lr0: 0.002
# ... and many more parameters
```

### Extraction Settings (`configs/extraction.yaml`)

```yaml
run_extraction: true
max_samples_per_half: null  # null for all samples
det_start_sec: 0.5
frame_shift: 0
class_names:
  - "player"  # Currently only player class available
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
- **Frame-label alignment preservation** from original training code
- **Configurable sampling** with max samples per half
- **Multi-season splitting** (train/val/test)
- **YOLO format conversion** with validation

### Training Pipeline
- **Configuration-driven training** with full YOLO parameter support
- **Automatic mixed precision** based on device capabilities
- **Progress logging** with grouped parameter display
- **Model evaluation and inference** support

### Logging & Monitoring
- **Structured logging** with configurable levels
- **File logging** support
- **Progress tracking** with detailed status updates
- **Error handling** with graceful fallbacks

## 🔄 Migration from Original Code

The original `train_detector.py` has been refactored into:

1. **Configuration files** replace hardcoded CONFIG section
2. **Device manager** replaces `select_training_device()` function
3. **Dataset extractor** replaces `batch_extract_dataset()` calls
4. **Trainer module** replaces inline YOLO training code
5. **Main orchestrator** replaces the original `main()` function

### Key Improvements

- ✅ **Modular architecture** with clear separation of concerns
- ✅ **Configuration-driven** parameters with CLI support
- ✅ **Better error handling** and logging
- ✅ **Code reusability** across different training scenarios
- ✅ **Maintainable structure** with focused modules
- ✅ **Preserved functionality** - exact same training behavior

## 📋 Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- PyYAML
- tqdm

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
# Extract small dataset for testing
python main.py --extract-only --max-samples-per-half 100

# Quick training test
python main.py --train-only --epochs 10 --batch 8

# Full training
python main.py --epochs 200
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

This modular architecture maintains all the functionality of the original training code while providing better organization, configuration management, and extensibility.