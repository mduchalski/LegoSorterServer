# TODO Switch to nvcr.io/nvidia/tensorrt
FROM nvcr.io/nvidia/tensorflow:22.10.1-tf2-py3

# OpenCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Python dependencies
ADD requirements.txt /LegoSorterServer/requirements.txt
RUN pip install -r /LegoSorterServer/requirements.txt

ADD . /LegoSorterServer/
WORKDIR /LegoSorterServer/
ENV PYTHONPATH="/LegoSorterServer"

# This downloads YOLOv3 model and dependencies
RUN python -c "from lego_sorter_server.analysis.detection.detectors.LegoDetectorProvider \
    import LegoDetectorProvider; LegoDetectorProvider.get_default_detector()"
