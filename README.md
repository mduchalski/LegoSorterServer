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

## Using Docker

Building the container:

```
docker build -t lego_sorter_server .
```

Running the container:
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 50051:50051 -v $PWD:/LegoSorterServer_host --rm lego_sorter_server python lego_sorter_server/
```

Running the container in interactive mode for debug/development, with local directory mounted as `/LegoSorterServer_host`:
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 50051:50051 --rm -it lego_sorter_server
```
*TODO: Come up with a nicer way to do this.* 

Running the TensorRT optimization:
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 50051:50051 -v $PWD:/LegoSorterServer_host --rm lego_sorter_server python lego_sorter_server/
```