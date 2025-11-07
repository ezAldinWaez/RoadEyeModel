# RoadEyeModel

Jupyter-based experiments and scripts for training and evaluating object detection models for road monitoring.

## Contents

### Experiments

Jupyter notebooks exploring different computer vision techniques:

- Image preprocessing (grayscale, binary conversion, histogram analysis)
- Foreground extraction and background subtraction
- Morphological operations for noise reduction
- Object detection and tracking evaluation

### Scripts

**preprocessing.py**

- Background extraction using median filtering
- Foreground segmentation via background subtraction
- Morphological operations (erosion, dilation)
- Processes both video files and image sequences

**prediction.py**

- YOLOv8 object detection inference
- Multi-object tracking with centroid tracking
- Trajectory smoothing using Savitzky-Golay filtering
- Speed and direction calculation from motion vectors
- Visualization with bounding boxes and trajectory lines

## Quick Usage

```python
from scripts.preprocessing import preprocessVideoFile
from scripts.predection import predectOnVideo

# Preprocess video
preprocessVideoFile('input.mp4', './input', './output')

# Run detection and tracking
predectOnVideo('./output/input.mp4', '../models/pretrained_e50.pt')
```

## Related

[RoadEyeApp](https://github.com/ezAldinWaez/RoadEyeApp) - Web application