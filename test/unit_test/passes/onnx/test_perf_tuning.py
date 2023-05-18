# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path
from test.unit_test.utils import get_onnx_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OrtPerfTuning
from olive.systems.local import LocalSystem


def test_ort_perf_tuning_pass():
    # setup
    local_system = LocalSystem()
    input_model = get_onnx_model()
    config = {"input_names": ["input"], "input_shapes": [[1, 1]]}
    p = create_pass_from_dict(OrtPerfTuning, config, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        local_system.run_pass(p, input_model, output_folder)
