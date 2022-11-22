import logging
import os
import time

import tensorflow as tf
import numpy as np

from typing import List

from tensorflow import keras
from PIL.Image import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple

KERAS_MODEL_PATH = 'lego_sorter_server/analysis/classification/models/keras_model'
TRT_MODEL_PATH = 'lego_sorter_server/analysis/classification/models/trt_model'

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class KerasClassifier(LegoClassifier):
    def __init__(self):
        super().__init__()
        self.model = None
        self.initialized = False
        self.size = (224, 224)

    @staticmethod
    def _trt_optimize_model(model_base):
        # See:
        # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/compiler/tensorrt/trt_convert.py

        model_base.save(KERAS_MODEL_PATH)

        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=KERAS_MODEL_PATH,
            precision_mode=trt.TrtPrecisionMode.FP16
        )
        converter.convert()
        # TODO add a build() stage to generate prebuild engines; this'll require some input data
        converter.save(output_saved_model_dir=TRT_MODEL_PATH)

        return tf.saved_model.load(TRT_MODEL_PATH)


    def load_model(self):
        model_base = keras.models.load_model(os.path.join(KERAS_MODEL_PATH, '447_classes.h5'))

        if os.getenv('LEGO_USE_TRT') == '1':
            self.model = self._trt_optimize_model(model_base)
        else:
            self.model = model_base

        self.initialized = True

    def predict(self, images: List[Image]) -> ClassificationResults:
        if not self.initialized:
            self.load_model()

        if len(images) == 0:
            return ClassificationResults.empty()

        images_array = []
        start_time = time.time()
        for img in images:
            transformed = Simple.transform(img, self.size[0])
            img_array = np.array(transformed)
            img_array = np.expand_dims(img_array, axis=0)
            images_array.append(img_array)
        processing_elapsed_time_ms = 1000 * (time.time() - start_time)

        if os.getenv('LEGO_USE_TRT') == '1':
            predictions = self.model(np.vstack(images_array).astype(np.float32))
        else:
            predictions = self.model(np.vstack(images_array))
        
        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        logging.info(f"[KerasClassifier] Preparing images took {processing_elapsed_time_ms} ms, "
                     f"when predicting took {predicting_elapsed_time_ms} ms.")

        indices = [int(np.argmax(values)) for values in predictions]
        classes = [self.class_names[index] for index in indices]
        scores = [float(prediction[index]) for index, prediction in zip(indices, predictions)]

        return ClassificationResults(classes, scores)
