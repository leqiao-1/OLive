# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
import transformers
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from onnxruntime.quantization.calibrate import CalibrationDataReader
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

disable_progress_bar()

class GlueCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        self.dataloader = create_dataloader(data_dir, batch_size)
        self.iter = iter(self.dataloader)

    def get_next(self):
        if self.iter is None:
            self.iter = iter(self.dataloader)
        try:
            batch = next(self.iter)
        except StopIteration:
            return None

        batch = {k: v.detach().cpu().numpy() for k, v in batch[0].items()}
        return batch

    def rewind(self):
        self.iter = None


def glue_calibration_reader(data_dir, batch_size=1):
    return GlueCalibrationDataReader(data_dir, batch_size=batch_size)


