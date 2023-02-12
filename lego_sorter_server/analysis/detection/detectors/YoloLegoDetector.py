import os
import threading
import time
import logging
import numpy
from pathlib import Path

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector

if os.getenv('LEGO_DETECTION_BACKEND') == 'tensorrt':
    from lego_sorter_server.analysis.detection.models.TrtDetectionModel import DetectionModel
else:
    from lego_sorter_server.analysis.detection.models.TorchDetectionModel import DetectionModel

class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class YoloLegoDetector(LegoDetector, metaclass=ThreadSafeSingleton):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "detection", "models", "yolo_model",
                                               "yolov5_medium_extended")):
        self.__initialized = False
        self.model_path = Path(model_path).absolute()

    def __initialize__(self):
        if self.__initialized:
            raise Exception("YoloLegoDetector already initialized")

        start_time = time.time()
        self.model = DetectionModel(self.model_path)
        elapsed_time = time.time() - start_time

        logging.info("Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True

    def detect_lego(self, image: numpy.ndarray) -> DetectionResults:
        if not self.__initialized:
            logging.info("YoloLegoDetector is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        logging.info("[YoloLegoDetector][detect_lego] Detecting bricks...")
        start_time = time.time()
        results = self.model(image)
        elapsed_time = 1000 * (time.time() - start_time)
        logging.info(f"[YoloLegoDetector][detect_lego] Detecting bricks took {elapsed_time} milliseconds")

        return results
