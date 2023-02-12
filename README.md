# LegoSorterServer

Lego Sorter Server provides methods for detecting and classifying Lego bricks.

## How to run
1. Download the repository
```commandline
git clone https://github.com/LegoSorter/LegoSorterServer.git
```
2. Download network models for detecting lego bricks
```commandline
wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/detection_models.zip
wget https://github.com/LegoSorter/LegoSorterServer/releases/download/1.2.0/classification_models.zip
```

3. Extract models
```commandline
unzip detection_models.zip -d ./LegoSorterServer/lego_sorter_server/analysis/detection/models
unzip classification_models.zip -d ./LegoSorterServer/lego_sorter_server/analysis/classification/models
```

4. Go to the root directory
```commandline
cd LegoSorterServer
```

5. Export *PYTHONPATH* environmental variable
```commandline
export PYTHONPATH=.
```

6. Run the server
```commandline
python lego_sorter_server
```

The server is now ready to handle requests. By default, the server listens on port *50051*

## Lego Sorter App
To test **Lego Sorter Server**, use the [Lego Sorter App](https://github.com/LegoSorter/LegoSorterApp), which is an application dedicated for this project.

## How to send a request (Optional)
**Lego Sorter Server** uses [gRPC](https://grpc.io/) to handle requests, the list of available methods is defined in `LegoSorterServer/lego_sorter_server/proto/LegoBrick.proto`.\
To call a method with your own client, look at [gRPC documentation](https://grpc.io/docs/languages/python/basics/#calling-service-methods)

## Docker and TensorRT support

There is a Docker container available for running the server on Nvidia GPUs. In addition to "standard" PyTorch and TensorFlow CUDA inference runtime, TensorRT use is also supported for both classification and detection NNs. This significantly speeds up the execution by leveraging optimizations specific to HW architecture for a particular Nvidia GPU/compute capability.

### Configuration

The following environement variables are exposed for configuration:
- `LEGO_DETECTION_BACKEND` - specifies detection inference engine
  - `tensorrt` - use TensorRT
  - default - use PyTorch CUDA runtime
- `LEGO_CLASSIFICATION_BACKEND` - specifies classification inference engine
  - `tensorrt` - use TensorRT
  - default - use TensorFlow CUDA runtime.

Note that when `tensorrt` backend is enabled for any task, then on server start TensorRT engines will be built. This can take a few minutes. To make those engines persistent, mount `/LegoSorterServer` in a named volume. Such volume should generally not be migrated across different machines/HW configurations, to get the most performence on a given system.

### Command reference

Start the container with TensorRT disabled:
```
docker run --gpus all -p 50051:50051 --rm \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	lego_sorter_server python lego_sorter_server/
```

Start the container with TensorRT enabled and a named volume mounted:
```
docker run --gpus all -p 50051:50051 --rm \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-e LEGO_CLASSIFICATION_BACKEND=tensorrt \
	-e LEGO_DETECTION_BACKEND=tensorrt \
	-v lego:/LegoSorterServer \
	lego_sorter_server python lego_sorter_server/
```

Remove an existing volume:
```
docker volume rm lego
```

Rebuilding the container:
```
docker build -t lego_sorter_server .
```
