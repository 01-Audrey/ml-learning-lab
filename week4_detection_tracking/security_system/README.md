# Security System - Object Detection & Tracking

A professional-grade security system for real-time object detection and tracking using YOLOv8 and DeepSORT.

## Features

- **Real-time Detection**: YOLOv8 for fast and accurate object detection
- **Multi-Object Tracking**: DeepSORT for persistent track IDs
- **Video Processing**: Process video files or live camera feeds
- **Optimized Performance**: 25-30 FPS on CPU, 50-60 FPS on GPU
- **Configurable**: YAML-based configuration system
- **Professional Logging**: Structured logging with file and console output
- **CLI Interface**: Easy-to-use command-line interface

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- YOLO v8
- DeepSORT

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd security_system
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLO model (if needed)

The YOLOv8n model will be downloaded automatically on first run.

## Usage

### Basic Usage

Process a video file:
```bash
python main.py input.mp4 -o output.mp4
```

### Live Camera Feed

Use webcam (camera index 0):
```bash
python main.py 0 --display
```

### Advanced Options
```bash
# Custom configuration
python main.py video.mp4 --config config/my_config.yaml

# Show live display window
python main.py video.mp4 --display

# Debug logging
python main.py video.mp4 --log-level DEBUG

# Console output only (no log file)
python main.py video.mp4 --no-log-file
```

### Command-Line Arguments
```
positional arguments:
  input                 Input video file or camera index (0 for webcam)

optional arguments:
  -h, --help            Show help message
  -o, --output OUTPUT   Output video file path
  -c, --config CONFIG   Configuration file (default: config/default.yaml)
  -d, --display         Show live display window
  --log-level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)
  --no-log-file         Disable log file
```

## Configuration

Edit `config/default.yaml` to customize behavior:
```yaml
# Model settings
model:
  yolo_model: "yolov8n.pt"
  confidence: 0.5
  device: "cpu"  # Options: "cpu", "cuda", "mps"

# Tracking parameters
tracking:
  max_age: 30      # Max frames to keep track alive
  n_init: 3        # Min detections before confirmed

# Video processing
video:
  target_resolution: [640, 480]  # Or null for original
  skip_frames: 0   # Process every Nth frame (0 = no skipping)
  save_output: true
```

## Project Structure
```
security_system/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py               # CLI entry point
│
├── config/               # Configuration files
│   └── default.yaml
│
├── src/                  # Source code
│   ├── config.py        # Configuration management
│   ├── models.py        # Model loading
│   ├── video_io.py      # Video input/output
│   ├── processor.py     # Processing pipeline
│   ├── utils.py         # Utility functions
│   └── logging_config.py # Logging setup
│
├── logs/                # Log files
├── output/              # Output videos
└── test_videos/         # Test data
```

## Performance

### Optimization Settings

For real-time performance on CPU:
```yaml
video:
  target_resolution: [640, 480]  # Reduce resolution
  skip_frames: 1                 # Process every other frame
```

### Expected Performance

| Configuration | Resolution | FPS (CPU) | FPS (GPU) |
|--------------|------------|-----------|-----------|
| Baseline     | 1920x1080  | 15-20     | 60-80     |
| Optimized    | 640x480    | 25-35     | 100+      |
| + Skip frames| 640x480    | 40-60     | 150+      |

## Testing

Run with test video:
```bash
# Create test video (if needed)
python -c "from src.utils import create_test_video; create_test_video('test.mp4')"

# Process test video
python main.py test.mp4 -o output.mp4 --display
```

## Output

Processed videos are saved to the `output/` directory with:
- Bounding boxes around detected objects
- Track IDs displayed
- Real-time FPS counter

Logs are saved to `logs/system.log` with:
- Processing statistics
- Detection counts
- Performance metrics
- Error messages

## Troubleshooting

### Common Issues

**ImportError: No module named 'src'**
- Make sure you're running from the project root directory
- Check that `src/__init__.py` exists

**CUDA out of memory**
- Reduce resolution in config: `target_resolution: [640, 480]`
- Use CPU instead: `device: "cpu"`

**Low FPS**
- Enable frame skipping: `skip_frames: 1`
- Reduce resolution
- Use GPU if available

**Video not opening**
- Check file path is correct
- Try different video codec
- Check OpenCV installation

## Development

### Adding New Features

1. Create new module in `src/`
2. Update configuration if needed
3. Update README
4. Test thoroughly

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions/classes
- Use logging instead of print statements

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- **YOLOv8**: Ultralytics
- **DeepSORT**: NWojke's deep_sort
- **OpenCV**: Computer vision library

## Contact

For questions or issues, please open a GitHub issue.

---

**Built for security and safety applications**
