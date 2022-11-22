FROM nvcr.io/nvidia/tensorflow:22.10.1-tf2-py3
ADD . /LegoSorterServer/
WORKDIR /LegoSorterServer/

# OpenCV dependencies
RUN apt install ffmpeg libsm6 libxext6 -y

RUN pip install -r /LegoSorterServer/requirements.txt

ENV PYTHONPATH="/LegoSorterServer"

# This downloads YOLOv3 model and dependencies
RUN python -c "from lego_sorter_server.analysis.detection.detectors.LegoDetectorProvider \
    import LegoDetectorProvider; LegoDetectorProvider.get_default_detector()"
