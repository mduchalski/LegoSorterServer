# LegoSorterServer

This application provides a processing server for a LEGO sorting [solution](https://github.com/LegoSorter/) developed in Gdansk University of Technology.
The program exposes a gRPC server intended for use by a corresponding [mobile application](https://github.com/LegoSorter/LegoSorterApp).

For brick detection and classification, deep CNN models are used.
By default, the inference is done using Torch and TensorFlow CUDA runtimes, respectively.
This fork additionally supports TensorRT-based runtime for both, which can significantly speed up the execution by leveraging optimizations specific to HW architecture for a particular NVIDIA GPU/compute capability.

It is recommended to use a [Docker container](https://hub.docker.com/r/mduchalski/lego_sorter_server) to run the application.
For standalone native setup, refer to [separate instructions](#native-setup).

## Quickstart

Prerequisites:
* Linux system with NVIDIA GPU - not tested with WSL/Windows
* Working Docker installation - see [official guide](https://docs.docker.com/get-docker/)
* NVIDIA Container Toolkit setup - see [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

To start the application:
```
docker run --gpus all -p 50051:50051 --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e LEGO_CLASSIFICATION_BACKEND=tensorrt \
    -e LEGO_DETECTION_BACKEND=tensorrt \
    -e CONVEYOR_LOCAL_ADDRESS=http://192.168.83.45:8000 \
    -e SORTER_LOCAL_ADDRESS=http://192.168.83.45:8001 \
    -v lego:/LegoSorterServer \
    -v $(realpath example_config.json):/LegoSorterServer/config.json \
    docker.io/mduchalski/lego_sorter_server \
    python lego_sorter_server/ -c config.json
```

This will start the server with default setup (same as fork origin), using TensorRT backend for inference.
The first run will take a few minutes as the TensorRT engines will be compiled.
This can be made persistent across restarts by mounting the app directory to a named volume, as above.
To enable the best performance on each system, such volume should generally not be migrated across different machines/HW configurations.

Refer to [configuration](#configration) for details and [command reference](#command-reference) for other commonly used commands.

## Configuration

The following environment variables are exposed for configuration:
* `LEGO_DETECTION_BACKEND` - specifies detection inference engine
  * `tensorrt` - use TensorRT
  * any other value - use PyTorch CUDA runtime
* `LEGO_CLASSIFICATION_BACKEND` - specifies classification inference engine
  * `tensorrt` - use TensorRT
  * any other value - use TensorFlow CUDA runtime
* `CONVEYOR_LOCAL_ADDRESS` - species motor control server address for the sorting module conveyor belt
  * valid address - will be used for motor control (e.g., http://192.168.83.45:8000)
  * not specified - communication with the motor control server will be silently omitted
* `SORTER_LOCAL_ADDRESS` - specifies motor control server address for the sorting module plow
  * valid address - will be used for motor control (e.g., http://192.168.83.45:8001)
  * not specified - communication with the motor control server will be silently omitted

Additionally, sorting configuration is specified in a JSON [configuration file](example_config.json):
* `bricks` - specifies plow position for desired brick classes
* `best_result_method` - specifies classification result aggregation method
  * `first` - use first result (fork origin default)
  * `max_score` - use result with the highest score
  * `mode` - use most common result
  * `min_inv_score` - use class for which product of inverse scores is the lowest

## Command reference

For local testing without use  
```
docker run --gpus all -p 50051:50051 --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e LEGO_CLASSIFICATION_BACKEND=tensorrt \
    -e LEGO_DETECTION_BACKEND=tensorrt \
    -v lego:/LegoSorterServer \
    docker.io/mduchalski/lego_sorter_server \
    python lego_sorter_server/
```

Start the container with TensorRT disabled, for local testing (no JSON/IP config):
```
docker run --gpus all -p 50051:50051 --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    docker.io/mduchalski/lego_sorter_server \
    python lego_sorter_server/
```

Remove an existing volume:
```
docker volume rm lego
```

Rebuild and push the update container image:
```
docker build  . -t docker.io/mduchalski/lego_sorter_server
docker push docker.io/mduchalski/lego_sorter_server
```

## Native setup

Prerequisite - [TensorFlow](https://www.tensorflow.org/install/pip) and [PyTorch](https://pytorch.org/get-started/locally/) set up for CUDA inference (see linked official guides).

Download network models:
```commandline
wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/detection_models.zip
wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/classification_models.zip
```

Extract models:
```commandline
unzip detection_models.zip -d ./lego_sorter_server/analysis/detection/models
unzip classification_models.zip -d ./lego_sorter_server/analysis/classification/models
```

Install required packages:
```commandline
pip3 install -r requirements.txt
```

Start the server:
```commandline
PYTHONPATH=. python3 lego_sorter_server
```
