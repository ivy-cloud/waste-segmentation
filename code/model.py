# -*- coding: utf-8 -*-
import cv2  # OpenCV for image/video processing
import torch
from QtFusion.models import Detector, HeatmapGenerator  # Detector base class from QtFusion
from class_name_list import Class_names  # Optional display-name mapping for classes
from ultralytics import YOLO  # YOLO model loader
from ultralytics.utils.torch_utils import select_device  # Device selection helper
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

ini_params = {
    "device": device,  # Inference device
    "conf": 0.3,  # Object confidence threshold
    "iou": 0.05,  # IoU threshold for NMS
    "classes": None,  # Class filter; None means no filtering
    "verbose": False,
}


def count_classes(det_info, class_names):
    """
    Count the number of each class in the detection info.

    :param det_info: List of detection info, each item is a list like [class_name, bbox, conf, class_id]
    :param class_names: List of all possible class names
    :return: A list with counts of each class
    """
    count_dict = {name: 0 for name in class_names}  # Count per class
    for info in det_info:  # Iterate detections
        class_name = info["class_name"]  # Class label
        if class_name in count_dict:  # Known class
            count_dict[class_name] += 1  # Increment count

    # Convert the dictionary to a list in the same order as class_names
    count_list = [count_dict[name] for name in class_names]
    return count_list


class Web_Detector(Detector):  # YOLO detector used by the web UI
    def __init__(self, params=None):  # Constructor
        super().__init__(params)  # Initialize base Detector
        self.model = None
        self.img = None  # Current image
        self.names = list(Class_names.values())  # Display names
        self.params = params if params else ini_params  # Runtime params

    def load_model(self, model_path):  # Load YOLO weights
        self.device = select_device(self.params["device"])  # Pick device
        # print(os.path.basename(model_path)[:3])
        if os.path.basename(model_path)[:3] == "seg":
            task = "segment"
        else:
            task = "segment"
        self.model = YOLO(model_path, task=task)
        names_dict = self.model.names  # Model class-name dict
        # Map to display names when available; otherwise keep original English labels
        self.names = [Class_names[v] if v in Class_names else v for v in names_dict.values()]
        self.model(
            torch.zeros(1, 3, *[self.imgsz] * 2).to(self.device).type_as(next(self.model.model.parameters()))
        )  # Warmup

    def preprocess(self, img):  # Preprocess input
        self.img = img  # Store original image
        return img  # Return processed image

    def predict(self, img):  # Run inference
        results = self.model(img, **ini_params)
        return results

    def postprocess(self, pred):  # Convert model output to dicts
        results = []  # Result list

        for res in pred[0].boxes:
            aim_id = 0
            for box in res:
                # Precompute and convert types
                class_id = int(box.cls.cpu())
                bbox = box.xyxy.cpu().squeeze().tolist()
                bbox = [int(coord) for coord in bbox]  # Integer bbox coords

                result = {
                    "class_name": self.names[class_id],  # Class name
                    "bbox": bbox,  # Bounding box
                    "score": box.conf.cpu().squeeze().item(),  # Confidence
                    "class_id": class_id,  # Class ID
                    "mask": pred[0].masks[aim_id].xy if pred[0].masks is not None else None,  # Mask polygon
                }
                results.append(result)  # Append detection

        # for seg in pred[0].masks:
        #     for mask in seg:
        #         print(mask)

        return results  # Return detections

    def set_param(self, params):
        self.params.update(params)
