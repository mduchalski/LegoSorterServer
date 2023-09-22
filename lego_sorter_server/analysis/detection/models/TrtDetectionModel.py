import os
import hashlib
import numpy as np

from cvu.detector import Detector
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults

class TrtDetectionResult:
    def __init__(self, result):
        pass

class DetectionModel:
    def __init__(self, model_path):
        # Conversion from PT to ONNX is a manual step, see:
        # https://github.com/ultralytics/yolov5/blob/master/export.py

        dtype = os.getenv('DETECTOR_TRT_DTYPE')
        if dtype == None:
            dtype = 'fp32'

        self.model = Detector(classes='lego', backend='tensorrt', weight=str(model_path) + '.onnx', dtype=dtype)

        # Run inference on dummy data to build the engine
        dummy_input = np.ones((640, 640, 3), dtype=np.uint8)
        self.model(dummy_input)

        engine_path = str(model_path) + '.engine'
        with open(engine_path, 'rb') as engine_file:
            self.hash = hashlib.sha256(engine_file.read()).hexdigest()
  
    def __call__(self, image):
        results = self.model(image)

        scores = np.array([result.confidence for result in results])
        classes = np.array([result.class_id for result in results]).astype(np.int64) + 1
        xyxy = np.array([result.bbox for result in results]) / 640
        boxes = np.array([[c[1], c[0], c[3], c[2]] for c in xyxy])

        return DetectionResults(detection_scores=scores, detection_classes=classes, detection_boxes=boxes)
