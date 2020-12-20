import random
import string
import time
from concurrent import futures

import numpy as np

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.LegoDetector import LegoDetector
from lego_sorter_server.images.queue.ImageProcessingQueue import ImageProcessingQueue
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage


class LegoDetectionRunner:
    def __init__(self, queue: ImageProcessingQueue, detector: LegoDetector, store: LegoImageStorage):
        self.queue = queue
        self.detector = detector
        self.storage = store
        # queue, detector and storage are not thread safe, so it limits the number of workers to one
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        print("[LegoDetectionRunner] Initialized\n")

    def start_detecting(self):
        print("[LegoDetectionRunner] Started processing the queue\n")
        self.executor.submit(self._process_queue)

    def stop_detecting(self):
        print("[LegoDetectionRunner] Processing is being terminated\n")
        self.executor.shutdown()

    def _process_queue(self):
        while True:
            if self.queue.len() == 0:
                print("Queue is empty. Waiting... ")
                time.sleep(1)
                continue
            image, lego_class = self.queue.next()
            prefix = self._get_random_hash() + "_"

            width, height = image.size
            image_resized, scale = DetectionUtils.resize(image, 640)
            detections = self.detector.detect_lego(np.array(image_resized))

            detected_counter = 0
            for i in range(100):
                if detections['detection_scores'][i] < 0.5:
                    break  # IF SORTED

                detected_counter += 1
                ymin, xmin, ymax, xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]
                if ymax >= height or xmax >= width:
                    continue
                image_new = image.crop([xmin, ymin, xmax, ymax])

                self.storage.save_image(image_new, lego_class, prefix)

            prefix = f'{detected_counter}_{prefix}'
            self.storage.save_image(image, 'original', prefix)

    @staticmethod
    def _get_random_hash(length=4):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))