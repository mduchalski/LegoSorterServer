import logging
import time

from typing import List, Tuple
from PIL.Image import Image

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.images.storage.LegoImageStorage import LegoImageStorage
from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig
from lego_sorter_server.sorter.LegoSorterController import LegoSorterController
from lego_sorter_server.sorter.ordering.SimpleOrdering import SimpleOrdering


class SortingProcessor:
    def __init__(self, brickCategoryConfig: BrickCategoryConfig):
        self.analysis_service: AnalysisService = AnalysisService()
        self.sorter_controller: LegoSorterController = LegoSorterController(brickCategoryConfig)
        self.ordering: SimpleOrdering = SimpleOrdering()
        self.storage: LegoImageStorage = LegoImageStorage()

        method_mapping = {
            'first': self.get_best_result_first,
            'max_score': self.get_best_result_max_score,
            'majority_vote': self.get_best_result_majority_vote,
            'prod_score': self.get_best_result_prod_score,
            'inv_prod_score': self.get_best_result_inv_prod_score,
            'sum_score': self.get_best_result_sum_score,
            'min_score': self.get_best_result_min_score,
            'med_score': self.get_best_result_med_score,
            'avg_score': self.get_best_result_avg_score
        }

        method = method_mapping.get(brickCategoryConfig.best_result_method)
        if method is None:
            raise ValueError(f'Unrecognized best_result_method: {brickCategoryConfig.best_result_method}')
        
        self.get_best_result = method

    def process_next_image(self, image: Image, save_image: bool = True):
        start_time = time.time()
        current_results = self._process(image)
        elapsed_ms = 1000 * (time.time() - start_time)

        logging.info(f"[SortingProcessor] Processing an image took {elapsed_ms} ms.")

        self.ordering.process_current_results(current_results, image_height=image.height)

        if save_image is True and len(current_results) > 0:
            start_time_saving = time.time()
            time_prefix = f"{int(start_time_saving * 10000) % 10000}"  # 10 seconds
            for key, value in self.ordering.get_current_state().items():
                bounding_box = value[0]
                cropped_image = DetectionUtils.crop_with_margin(image, *bounding_box)
                self.storage.save_image(cropped_image, str(key), time_prefix)
            self.storage.save_image(image, "original_sorter", time_prefix)
            logging.info(f"[SortingProcessor] Saving images took {1000 * (time.time() - start_time_saving)} ms.")

        while self.ordering.get_count_of_results_to_send() > 0:
            best_result = self._send_results_to_controller()
            if best_result != None:
                    return {-1: best_result}

        return self.ordering.get_current_state()

    def _send_results_to_controller(self):
        processed_brick = self.ordering.pop_first_processed_brick()

        if len(processed_brick) == 0:
            return None

        best_result = self.get_best_result(processed_brick)
        logging.info(f"[SortingProcessor] Got the best result {best_result}. Returning the results...")
        self.sorter_controller.on_brick_recognized(best_result)
        return best_result

    def _process(self, image: Image) -> List[Tuple]:
        """
        Returns a list of recognized bricks ordered by the position on the belt - ymin desc
        """
        results = self.analysis_service.detect_and_classify(image, detection_threshold=0.8)

        detected_count = len(results[0].detection_classes)
        if detected_count == 0:
            return []

        logging.info(f"[SortingProcessor] Detected a lego brick, processing...")

        if detected_count > 1:
            logging.warning(f"[SortingProcessor] More than one brick detected '(detected_count = {detected_count}), "
                            f"there should be only one brick on the tape at the same time.")

        zipped_results = list(zip(results[0].detection_boxes,
                                  results[1].classification_classes,
                                  results[1].classification_scores))

        return self.order_by_bounding_box_position(zipped_results)

    def start_machine(self):
        self.sorter_controller.run_conveyor()

    def stop_machine(self):
        self.sorter_controller.stop_conveyor()

    def set_machine_speed(self, speed: int):
        self.sorter_controller.set_machine_speed(speed)

    @staticmethod
    def order_by_bounding_box_position(zipped_results: List[Tuple[Tuple, str, float]]) -> List[Tuple]:
        # sort by ymin
        return sorted(zipped_results, key=lambda res: res[0][0], reverse=True)

    @staticmethod
    def get_best_result_first(results):
        return results[0]

    @staticmethod
    def get_best_result_max_score(results):
        return max(results, key=lambda res: res[2])

    @staticmethod
    def get_best_result_majority_vote(results):
        label_scores = {}
        for _, label, _ in results:
            if label in label_scores:
                label_scores[label] += 1
            else:
                label_scores[label] = 1

        best_label = max(label_scores, key=lambda l: label_scores[l])
        return next(result for result in results if result[1] == best_label)

    @staticmethod
    def get_best_result_prod_score(results):
        label_scores = {}
        for _, label, score in results:
            if label in label_scores:
                label_scores[label] *= score
            else:
                label_scores[label] = score

        best_label = max(label_scores, key=lambda l: label_scores[l])
        return next(result for result in results if result[1] == best_label)

    @staticmethod
    def get_best_result_inv_prod_score(results):
        label_scores = {}
        for _, label, score in results:
            if label in label_scores:
                label_scores[label] *= (1-score)
            else:
                label_scores[label] = (1-score)

        best_label = min(label_scores, key=lambda l: label_scores[l])
        return next(result for result in results if result[1] == best_label)

    @staticmethod
    def get_best_result_sum_score(results):
        label_scores = {}
        for _, label, score in results:
            if label in label_scores:
                label_scores[label] += score
            else:
                label_scores[label] = score

        best_label = max(label_scores, key=lambda l: label_scores[l])
        return next(result for result in results if result[1] == best_label)

    @staticmethod
    def get_best_result_min_score(results):
        label_scores = {}
        for _, label, score in results:
            if label not in label_scores or score < label_scores[label]:
                label_scores[label] = score

        best_label = max(label_scores, key=lambda l: label_scores[l])
        return next(result for result in results if result[1] == best_label)

    @staticmethod
    def get_best_result_med_score(results):
        label_scores = {}
        for _, label, score in results:
            if label in label_scores:
                label_scores[label].append(score)
            else:
                label_scores[label] = [score]

        best_label = max(label_scores, key=lambda l: sorted(label_scores[l])[len(label_scores[l])//2])
        return next(result for result in results if result[1] == best_label)

    @staticmethod
    def get_best_result_avg_score(results):
        label_scores = {}
        for _, label, score in results:
            if label in label_scores:
                label_scores[label].append(score)
            else:
                label_scores[label] = [score]

        best_label = max(label_scores, key=lambda l: sum(label_scores[l]) / len(label_scores[l]))
        return next(result for result in results if result[1] == best_label)
