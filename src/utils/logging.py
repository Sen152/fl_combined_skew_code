"""实验日志工具"""

import json
import os
from datetime import datetime


class ExperimentLogger:
    """记录实验结果"""

    def __init__(self, save_dir: str, exp_name: str):
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.records = []
        os.makedirs(save_dir, exist_ok=True)

    def log_round(self, round_num: int, metrics: dict):
        """记录每轮结果"""
        record = {"round": round_num, "timestamp": datetime.now().isoformat()}
        record.update(metrics)
        self.records.append(record)

    def save(self):
        """保存到JSON"""
        path = os.path.join(self.save_dir, f"{self.exp_name}.json")
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        return path
