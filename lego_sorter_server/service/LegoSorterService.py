import hashlib
import logging
import time
import os

from io import BytesIO

from lego_sorter_server.generated import LegoSorter_pb2_grpc
from lego_sorter_server.generated.LegoSorter_pb2 import SorterConfiguration, ListOfBoundingBoxesWithIndexes, \
    BoundingBoxWithIndex
from lego_sorter_server.generated.Messages_pb2 import ImageRequest, BoundingBox, Empty
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils
from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig
from lego_sorter_server.sorter.SortingProcessor import SortingProcessor


class LegoSorterService(LegoSorter_pb2_grpc.LegoSorterServicer):

    def __init__(self, brickCategoryConfig: BrickCategoryConfig):
        self.sortingProcessor = SortingProcessor(brickCategoryConfig)

    def processNextImage(self, request: ImageRequest, context) -> ListOfBoundingBoxesWithIndexes:
        start_time = time.time()
        logging.info("[LegoSorterService] Got an image request. Processing...")
        self.sortingProcessor.analysis_service.logger.update('processing_start_time', start_time)

        m = hashlib.sha256()
        print(type(request.image))
        m.update(request.image)
        self.sortingProcessor.analysis_service.logger.update('recv_hash', m.hexdigest())

        image = ImageProtoUtils.prepare_image(request)
        current_state = self.sortingProcessor.process_next_image(image)

        response = self._prepare_response_from_sorter_state(current_state=current_state)
        end_time = time.time()
        elapsed_milliseconds = int(1000 * (end_time - start_time))
        self.sortingProcessor.analysis_service.logger.update('processing_stop_time', end_time, save_log=True)
        logging.info(f"[LegoSorterService] Processing the request took {elapsed_milliseconds} milliseconds.")
        
        return response

    def startMachine(self, request: Empty, context):
        self.sortingProcessor.start_machine()

        return Empty()

    def stopMachine(self, request: Empty, context):
        self.sortingProcessor.stop_machine()

        return Empty()

    def getConfiguration(self, request, context) -> SorterConfiguration:
        return super().getConfiguration(request, context)

    def updateConfiguration(self, request: SorterConfiguration, context):
        self.sortingProcessor.set_machine_speed(request.speed)

        return Empty()

    @staticmethod
    def _prepare_response_from_sorter_state(current_state: dict) -> ListOfBoundingBoxesWithIndexes:
        bbs_with_indexes = []
        for key, value in current_state.items():
            bb = BoundingBox()
            bb.ymin, bb.xmin, bb.ymax, bb.xmax = value[0]
            bb.label = value[1]
            bb.score = value[2]

            bb_index = BoundingBoxWithIndex()
            bb_index.bb.CopyFrom(bb)
            bb_index.index = key

            bbs_with_indexes.append(bb_index)

        response = ListOfBoundingBoxesWithIndexes()
        response.packet.extend(bbs_with_indexes)

        return response
