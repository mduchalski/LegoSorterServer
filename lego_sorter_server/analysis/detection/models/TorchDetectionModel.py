import torch

class DetectionModel:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        if torch.cuda.is_available():
            self.model.cuda()
    
    def __call__(self, image):
        return self.model([image], size=image.shape[0])