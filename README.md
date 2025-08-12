# 3D Object Detection and Tracking for Autonomous Vehicles

An advanced multi-modal perception system for autonomous vehicles that integrates **YOLOv8 object detection**, **stereo depth estimation**, and **DeepSORT tracking** to provide robust 3D object detection and tracking capabilities. This project addresses the critical challenge of accurate 3D object detection and tracking in autonomous vehicles by combining the strengths of modern deep learning architectures with classical computer vision techniques. The system processes stereo camera inputs to detect, localize, and track objects in 3D space with high accuracy and real-time performance.

### Key Achievements

- **72.4% mAP@0.5** across all object classes
- **89.9% mAP** for vehicle detection
- **31 FPS** real-time performance
- **32.3ms** processing time per frame

## Features

### Core Capabilities
- **Fine-tuned YOLOv8**: Optimized for autonomous driving scenarios using KITTI dataset
- **Stereo Depth Estimation**: Real-time depth computation using Semi-Global Block Matching (SGBM)
- **Multi-Object Tracking**: Consistent object tracking with DeepSORT algorithm
- **3D Bounding Boxes**: Accurate 3D object localization and dimension estimation
- **Real-time Processing**: Optimized for autonomous vehicle applications

### Technical Highlights
- Custom KITTI-to-YOLO dataset conversion pipeline
- Robust stereo calibration and depth estimation
- Appearance-based tracking with deep features
- Comprehensive evaluation metrics and visualization tools
- GPU acceleration support

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3060 or better)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 50GB free space for dataset and models

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (for GPU acceleration)
- **Operating System**: Linux (Ubuntu 18.04+), Windows 10+, or macOS

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/3d-object-detection-autonomous-vehicles.git
cd 3d-object-detection-autonomous-vehicles
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n 3d-detection python=3.8
conda activate 3d-detection

# Or using venv
python -m venv 3d-detection
source 3d-detection/bin/activate  # Linux/Mac
# or
3d-detection\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download KITTI Dataset
The system uses the KITTI dataset for training and evaluation. The dataset will be automatically downloaded using KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("klemenko/kitti-dataset")
```

## Project Structure

```
3d-object-detection/
├── 3D_Object_Detection.ipynb          # Main Jupyter notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                           # License file
├── docs/                            # Documentation
│   └── project_report.pdf           # Detailed project report
├── src/                             # Source code modules
│   ├── __init__.py
│   ├── detection/                   # Object detection modules
│   │   ├── __init__.py
│   │   ├── yolo_detector.py        # YOLOv8 detection
│   │   └── model_training.py       # Training utilities
│   ├── depth/                      # Depth estimation modules
│   │   ├── __init__.py
│   │   ├── stereo_matcher.py       # Stereo matching
│   │   └── calibration.py          # Camera calibration
│   ├── tracking/                   # Object tracking modules
│   │   ├── __init__.py
│   │   ├── deepsort_tracker.py     # DeepSORT implementation
│   │   └── kalman_filter.py        # Kalman filtering
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading utilities
│   │   ├── visualization.py        # Visualization tools
│   │   └── evaluation.py           # Evaluation metrics
│   └── integration/                # System integration
│       ├── __init__.py
│       └── perception_system.py    # Main system class
├── data/                           # Data directory
│   ├── raw/                       # Raw KITTI data
│   ├── processed/                 # Processed datasets
│   └── models/                    # Trained models
├── configs/                       # Configuration files
│   ├── dataset.yaml              # YOLO dataset configuration
│   └── training_config.py        # Training parameters
├── scripts/                       # Utility scripts
│   ├── download_data.py          # Data download script
│   ├── train_model.py            # Model training script
│   └── evaluate_model.py         # Evaluation script
├── results/                       # Results and outputs
│   ├── metrics/                  # Evaluation metrics
│   ├── visualizations/           # Output visualizations
│   └── logs/                     # Training logs
└── tests/                        # Unit tests
    ├── __init__.py
    ├── test_detection.py
    ├── test_depth.py
    └── test_tracking.py
```

## Usage

### Quick Start with Jupyter Notebook

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook 3D_Object_Detection.ipynb
   ```

2. **Run the notebook cells sequentially** to:
   - Download and setup KITTI dataset
   - Convert KITTI annotations to YOLO format
   - Train the YOLOv8 model
   - Perform stereo depth estimation
   - Integrate detection, depth, and tracking

### Command Line Usage

#### 1. Data Preparation
```bash
python scripts/download_data.py --dataset kitti --output data/raw/
```

#### 2. Model Training
```bash
python scripts/train_model.py \
    --data configs/dataset.yaml \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640
```

#### 3. Model Evaluation
```bash
python scripts/evaluate_model.py \
    --model data/models/yolov8_kitti.pt \
    --data data/processed/test/ \
    --output results/
```

#### 4. Run Integrated System
```python
from src.integration.perception_system import IntegratedPerceptionSystem

# Initialize the system
system = IntegratedPerceptionSystem(
    yolo_model_path="data/models/yolov8_kitti.pt"
)

# Process stereo image pair
left_img = cv2.imread("path/to/left_image.png")
right_img = cv2.imread("path/to/right_image.png")
calib_file = "path/to/calibration.txt"

# Get 3D detections and tracking
results = system.process_frame_pair(left_img, right_img, calib_file)
```

## Configuration

### Dataset Configuration (configs/dataset.yaml)
```yaml
path: ./data/processed/kitti_yolo
train: images/train
val: images/val
nc: 6  # number of classes
names: ['Vehicle', 'Truck', 'Pedestrian', 'Cyclist', 'Tram', 'Misc']
```

### Training Configuration
Key parameters in `configs/training_config.py`:
- **Image size**: 640x640 pixels
- **Batch size**: 16 (adjust based on GPU memory)
- **Epochs**: 100 (with early stopping)
- **Learning rate**: 0.01 (with cosine annealing)
- **Augmentation**: Enabled (rotation, scaling, color jittering)

## Performance Metrics

### Detection Performance (KITTI Test Set)
| Class | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| Vehicle | 89.9% | 92.1% | 87.3% |
| Pedestrian | 78.5% | 81.2% | 75.8% |
| Cyclist | 65.3% | 68.7% | 62.1% |
| Truck | 71.2% | 74.5% | 68.9% |
| **Overall** | **72.4%** | **79.1%** | **73.5%** |

### System Performance
- **Processing Speed**: 31 FPS (32.3ms per frame)
- **Depth Accuracy**: RMSE < 2.5m for objects within 50m
- **Tracking Stability**: 95.2% ID consistency across sequences

### Component Breakdown
| Component | Processing Time | Percentage |
|-----------|----------------|------------|
| YOLOv8 Detection | 3.0ms | 9.3% |
| Stereo Depth Estimation | 26.5ms | 82.0% |
| DeepSORT Tracking | 2.8ms | 8.7% |
| **Total** | **32.3ms** | **100%** |

## Technical Details

### System Architecture
The system follows a modular pipeline design:

1. **Input Processing**: Stereo image pair and calibration data
2. **2D Detection**: Fine-tuned YOLOv8 processes left camera image
3. **Depth Estimation**: SGBM algorithm computes dense depth map
4. **3D Projection**: 2D detections projected to 3D using depth information
5. **Object Tracking**: DeepSORT maintains object identities across frames
6. **Output Generation**: 3D bounding boxes with tracking IDs

### Key Algorithms
- **Object Detection**: YOLOv8 with anchor-free detection
- **Stereo Matching**: Semi-Global Block Matching (SGBM)
- **Tracking**: DeepSORT with Kalman filtering and appearance features
- **3D Estimation**: Perspective projection with camera calibration

### Optimization Techniques
- **GPU Acceleration**: CUDA-optimized operations
- **Memory Management**: Efficient batch processing
- **Model Pruning**: Reduced model size for deployment
- **Pipeline Parallelization**: Concurrent processing stages

### Research Papers
This project builds upon several key research areas:
- Object Detection: YOLOv8 architecture and training techniques
- Stereo Vision: Classical and deep learning-based stereo matching
- Multi-Object Tracking: Kalman filtering and appearance-based methods


## Acknowledgments

- **KITTI Dataset**: Vision meets Robotics team for providing the dataset
- **Ultralytics**: For the excellent YOLOv8 implementation
- **DeepSORT**: Wojke et al. for the tracking algorithm
- **OpenCV**: For computer vision utilities
- **PyTorch**: For the deep learning framework

## Roadmap

### Current Version (v1.0)
- ✅ YOLOv8 fine-tuning for KITTI dataset
- ✅ Stereo depth estimation pipeline
- ✅ DeepSORT multi-object tracking
- ✅ 3D bounding box generation
- ✅ Comprehensive evaluation framework

### Upcoming Features
- 🔄 Real-time video processing interface
- 🔄 Deep learning-based stereo matching
- 🔄 Temporal fusion for improved accuracy
- 🔄 Multi-modal sensor fusion (camera + LiDAR)
- 🔄 End-to-end trainable architecture

---
