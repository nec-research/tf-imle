# -*- coding: utf-8 -*-

import os
from tensorboardX import SummaryWriter
import numpy as np
from collections import defaultdict

from maprop.decorators import input_to_numpy


class Logger:

    valid_outputs = ["tensorboard", "stdout", "csv"]

    logging_dir = None
    summary_writer = None
    step_for_key = defaultdict(lambda: 0)  # initialize step to 1 for every key
    default_output = None

    def __init__(self, scope, subdir=False, default_output=None):
        self.scope = scope
        self.subdir = subdir
        self.default_output = default_output or Logger.default_output

        if self.subdir:
            self.local_logging_dir = os.path.join(self.logging_dir, self.scope)
            self.local_summary_writer = SummaryWriter(os.path.join(self.local_logging_dir, "events"))

    @classmethod
    def configure(cls, logging_dir, default_output):
        cls.logging_dir = logging_dir
        cls.summary_writer = SummaryWriter(os.path.join(cls.logging_dir, "events"), flush_secs=30)
        if default_output not in cls.valid_outputs:
            raise NotImplementedError(f"{default_output} is not a valid output")
        else:
            cls.default_output = default_output

    def infer_datatype(self, data):
        if np.isscalar(data):
            return "scalar"
        elif isinstance(data, np.ndarray):
            if data.ndim == 0:
                return "scalar"
            elif data.ndim == 1:
                if data.size == 1:
                    return "scalar"
                if data.size > 1:
                    return "histogram"
            elif data.ndim == 2:
                return "image"
            elif data.ndim == 3:
                return "image"
            else:
                raise NotImplementedError("Numpy arrays with more than 2 dimensions are not supported")
        else:
            raise NotImplementedError(f"Data type {type(data)} not understood.")

    @input_to_numpy
    def log(self, data, key=None, data_type=None, to_tensorboard=None, to_stdout=None, to_csv=None):
        if data_type is None:
            data_type = self.infer_datatype(data)

        output_callables = []
        if to_tensorboard or (to_tensorboard is None and self.default_output == "tensorboard"):
            output_callables.append(self.to_tensorboard)
        if to_stdout or (to_stdout is None and self.default_output == "stdout"):
            output_callables.append(self.to_stdout)
        if to_csv or (to_csv is None and self.default_output == "csv"):
            output_callables.append(self.to_csv)

        for output_callable in output_callables:
            output_callable(key, data_type, data)

    def to_tensorboard(self, key, data_type, data):
        if key is None:
            raise ValueError("Logging to tensorboard requires a valid key")

        if self.subdir:
            summary_writer = self.local_summary_writer
        else:
            summary_writer = self.summary_writer

        step = self.step_for_key[key]

        self.step_for_key[key] += 1

        if data_type == "scalar":
            data_specific_writer_callable = summary_writer.add_scalar
        elif data_type == "histogram":
            data_specific_writer_callable = summary_writer.add_histogram
        elif data_type == "image":
            data_specific_writer_callable = summary_writer.add_image
        else:
            raise NotImplementedError(f"Summary writer does not support type {data_type}")

        data_specific_writer_callable(self.scope + "/" + key, data, step)

    def to_stdout(self, key, data_type, data):
        # if not data_type == "scalar":
        #    raise NotImplementedError("Only data type 'scalar' supported for stdout output")

        print(f"[{self.scope}] {key}: {data}")

    def to_csv(self, key, data_type, data):
        raise NotImplementedError("CSV output is not implemented, yet")

    def __del__(self, *args, **kwargs):
        if self.summary_writer is not None:
            self.summary_writer.close()
        self.summary_writer = None
