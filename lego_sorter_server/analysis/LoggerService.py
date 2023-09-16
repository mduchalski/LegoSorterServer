import os
import csv

from lego_sorter_server.analysis.classification.models.TrtClassificationModel import ClassificationModel as TrtClassificationModel

LOG_FIELDS = ['image_idx', 'processing_start_time', 'processing_stop_time', 'classify_start_time', 'classify_end_time', 'detect_start_time', 'detect_end_time', 'recv_hash']
PRINT_CONFIG = ['LOG_FILENAME', 'LEGO_DETECTION_BACKEND', 'LEGO_CLASSIFICATION_BACKEND', 'CLASSIFIER_TRTEXEC_FLAGS']

class LoggerService:

    def __init__(self, classifier):
        self._clear_log()

        self.log_name = os.getenv('LOG_FILENAME')
        self.image_idx = 0
        
        if self.log_name == None:
            return

        with open(self.log_name, 'w') as log_file:
            for cfg in PRINT_CONFIG:
                log_file.write(f'# {cfg} = {os.getenv(cfg)}\n')

            if isinstance(classifier.model, TrtClassificationModel) == True:
                log_file.write(f'# Classifier engine hash = {classifier.model.hash}\n')

            writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
            writer.writeheader()

    def _clear_log(self):
        self.log = {}
    
    def update(self, key, value, save_log=False):
        if key in LOG_FIELDS:
            self.log[key] = value

        if save_log == False or self.log_name == None:
            return

        self.log['image_idx'] = self.image_idx
        self.image_idx += 1

        with open(self.log_name, 'a') as log_file:
            writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
            writer.writerow(self.log)

        self._clear_log()
