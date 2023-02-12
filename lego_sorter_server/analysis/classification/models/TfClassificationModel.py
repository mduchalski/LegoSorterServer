import numpy as np
import tensorflow as tf

from tensorflow import keras

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class ClassificationModel:
    def __init__(self, model_path):
        self.model = keras.models.load_model(str(model_path) + '.h5')
    
    def __call__(self, images):
        return self.model(images)
