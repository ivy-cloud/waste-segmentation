import os
import torch
import yaml
from ultralytics import YOLO  # Import YOLO model
import matplotlib

matplotlib.use("TkAgg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":  # Run only when this module is executed directly
    workers = 4
    # Adjust batch size based on GPU/CPU memory; lower it if you run out of memory
    batch = 8
    device = "0" if torch.cuda.is_available() else "cpu"

    data_path = "./datasets/new_wastes/data.yaml"

    # Load YAML / pretrained weights while keeping class order
    model = YOLO("yolo11s-seg.pt")  # build a new model from YAML

    results = model.train(  # Start training
        data=data_path,  # Path to the training data config
        device=device,  # Auto-select GPU/CPU for training
        workers=workers,  # Number of data-loading workers
        imgsz=640,  # Input image size 640x640
        epochs=1,  # Number of training epochs
        batch=batch,  # Batch size
    )
