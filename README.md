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

Typical Ubuntu Docker setup (reference only - **highly recommended** to refer to [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)):
```commandline
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

To start the application:
```commandline
docker run --gpus all --network=host --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e LEGO_CLASSIFICATION_BACKEND=tensorrt \
    -e LEGO_DETECTION_BACKEND=tensorrt \
    -e CONVEYOR_LOCAL_ADDRESS=http://192.168.83.45:8000 \
    -e SORTER_LOCAL_ADDRESS=http://192.168.83.45:8001 \
    -v lego:/LegoSorterServer \
    docker.io/mduchalski/lego_sorter_server \
    python lego_sorter_server/ -c example_config.json
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
* `CLASSIFIER_TRTEXEC_FLAGS` - flags used when building TensorRT engine for classfication model
  * Refer to [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-ovr) for details
* `LOG_FILENAME` - filename of CSV performance log file
  * not specified - no log will be saved
* `CLASSIFIER_LAYER_INFO_PATH` - filename of JSON file to dump classifier TensorRT engine layer info to
  * not specified - layer info won't be exported
* `DETECTOR_TRT_DTYPE` - TensorRT datatype for detection
  * `fp16` - use float16
  * `fp32`, not spacified - use float32
* `BEST_RESULT_METHOD` - specifies multi-view classification method (result aggreagtion)
  * `first` - use first result (fork origin default)
  * `max_score` - use result with the highest score
  * `majority_vote` - use most common result
  * `prod_score` - use class with the highest product of scores
  * `inv_prod_score` - use class with the lowest product of inverted scores
  * `sum_score` - use class with the highest sum of scores
  * `min_score` - use class with the highest minumum score
  * `med_score` - use class with the highest median score
  * `avg_score` - use class with the highest average score

Additionally, sorting configuration is specified in a JSON [configuration file](example_config.json):
* `bricks` - specifies plow position for desired brick classes


## Command reference

Launch using custom mounted JSON config:
```
docker run --gpus all --network=host --rm \
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

Launch with legacy CUDA runtime:
```
docker run --gpus all --network=host --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e CONVEYOR_LOCAL_ADDRESS=http://192.168.83.45:8000 \
    -e SORTER_LOCAL_ADDRESS=http://192.168.83.45:8001 \
    -v lego:/LegoSorterServer \
    docker.io/mduchalski/lego_sorter_server \
    python lego_sorter_server/ -c config.json
```

Launch for local testing, with logging enabled and w/o motor controller communication:
```
docker run --gpus all --network=host --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e LEGO_CLASSIFICATION_BACKEND=tensorrt \
    -e LEGO_DETECTION_BACKEND=tensorrt \
    -e LOG_FILENAME=/mnt/logs/recv.csv \
    -v lego:/LegoSorterServer \
    -v $(realpath ./scripts/logs/):/mnt/logs \
    docker.io/mduchalski/lego_sorter_server \
    python lego_sorter_server/ -c example_config.json
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
