import os
import csv

LOG_FIELDS = ['image_idx', 'processing_start_time', 'processing_stop_time', 'classify_start_time', 'classify_end_time', 'detect_start_time', 'detect_end_time', 'recv_hash']

class LoggerService:

    def __init__(self):
        self._clear_log()

        self.log_name = os.getenv('LOG_FILENAME')
        self.image_idx = 0
        
        if self.log_name == None:
            return

        with open(self.log_name, 'w') as log_file:
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
