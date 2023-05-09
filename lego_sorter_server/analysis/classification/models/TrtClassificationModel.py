import os
import onnx
import tf2onnx
import numpy as np
import tensorrt as trt
import tensorflow as tf
import subprocess as sp
import pycuda.driver as cuda

from tensorflow import keras

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class ClassificationModel:
    def __init__(self, model_path):
        if not os.path.isfile(str(model_path) + '.engine'):
            tf_model = keras.models.load_model(str(model_path) + '.h5')

            # Convert to ONNX
            onnx_model, _ = tf2onnx.convert.from_keras(tf_model)
            onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
            onnx.save_model(onnx_model, str(model_path) + '.onnx')


            # Run TensorRT optimization
            # TODO expose flags override/extra through envar
            sp.check_call([ 'trtexec',
                        f'--onnx={str(model_path) + ".onnx"}',
                        f'--saveEngine={str(model_path) + ".engine"}',
                        '--inputIOFormats=fp16:chw', '--outputIOFormats=fp16:chw', '--fp16'])

        self._cuda_setup(str(model_path) + '.engine')

    def __call__(self, images):
        output = []
        for image in images:
            self.cuda_driver_context.push()
            cuda.memcpy_htod_async(self.d_input, image.astype(np.float16), self.stream)
            self.context.execute_async_v2(self.bindings, self.stream.handle, None)
            cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
            self.stream.synchronize()
            self.cuda_driver_context.pop()
            output.append(self.output.copy())
        return output

    def _cuda_setup(self, engine_path):
        cuda.init()
        device = cuda.Device(0)
        self.cuda_driver_context = device.make_context()

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        with open(engine_path, 'rb') as fp:
            engine = runtime.deserialize_cuda_engine(fp.read())    
        self.context = engine.create_execution_context()

        # TODO parametrize sizes
        input = np.empty((1, 224, 224, 3), dtype=np.float16)
        self.d_input = cuda.mem_alloc(1 * input.nbytes)
        self.output = np.empty((447,), dtype=np.float16)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()
