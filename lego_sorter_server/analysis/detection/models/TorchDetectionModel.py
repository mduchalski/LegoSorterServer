import torch
import numpy as np

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults


class DetectionModel:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path) + '.pt')
        if torch.cuda.is_available():
            self.model.cuda()
    
    def __call__(self, image):
        results = self.model([image], size=image.shape[0])

        image_predictions = results.xyxyn[0].cpu().numpy()
        scores = image_predictions[:, 4]
        classes = image_predictions[:, 5].astype(np.int64) + 1
        boxes = np.array([[c[1], c[0], c[3], c[2]] for c in image_predictions[:, :4]])

        return DetectionResults(detection_scores=scores, detection_classes=classes, detection_boxes=boxes)
