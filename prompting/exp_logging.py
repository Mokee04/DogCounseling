import logging, os, glob, json
import pandas as pd
import numpy as np
from datetime import datetime
from threading import Lock

class Logging:
    def __init__(self, save_dir: str, exp_iteration: str = ''):
        self.save_dir = save_dir
        self.exp_iteration = exp_iteration
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)
        self._lock = Lock()

    def log_row(
        self,
        event_name: str = '',
        timestamp: str = '',
        info: str = '',
        data: dict | None = None,
        additional_info: dict | None = None,
        **fields
    ) -> dict:
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            'event_name': event_name,
            'timestamp': timestamp,
            'info': info,
            **fields,
            'data': json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else (data or ''),
            'additional_info': json.dumps(additional_info, ensure_ascii=False) if isinstance(additional_info, dict) else (additional_info or '')
        }
        return row

    def save_logging(self, log_row: dict):
        header = not os.path.exists(self.save_dir) or os.path.getsize(self.save_dir) == 0
        df = pd.DataFrame([log_row])
        with self._lock:
            df.to_csv(self.save_dir, mode='a', index=False, header=header)