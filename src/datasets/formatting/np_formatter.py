# Copyright 2020 The HuggingFace Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from collections.abc import Mapping

import numpy as np
import pyarrow as pa

from .. import config
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter


class NumpyFormatter(TensorFormatter[Mapping, np.ndarray, Mapping]):
    def __init__(self, features=None, token_per_repo_id=None, **np_array_kwargs):
        super().__init__(features=features, token_per_repo_id=token_per_repo_id)
        self.np_array_kwargs = np_array_kwargs

    def _consolidate(self, column):
        if isinstance(column, list):
            # Fast path: check if all entries are ndarray with same shape and dtype.
            # Use an explicit loop and locals to avoid generator and repeated attribute lookups.
            if column:
                first = column[0]
                if isinstance(first, np.ndarray):
                    first_shape = first.shape
                    first_dtype = first.dtype
                    all_same = True
                    for x in column:
                        if not isinstance(x, np.ndarray) or x.shape != first_shape or x.dtype != first_dtype:
                            all_same = False
                            break
                    if all_same:
                        return np.stack(column)
            # Fallback: create an object-dtype array and assign elements
            out = np.empty(len(column), dtype=object)
            out[:] = column
            return out
        return column

    def _tensorize(self, value):
        if isinstance(value, (str, bytes, type(None))):
            return value

        # Handle numpy character arrays and numpy scalars early.
        if isinstance(value, np.ndarray):
            dtype = value.dtype
            if np.issubdtype(dtype, np.character):
                return value
            if np.issubdtype(dtype, np.integer):
                default_dtype = {"dtype": np.int64}
            elif np.issubdtype(dtype, np.floating):
                default_dtype = {"dtype": np.float32}
            else:
                default_dtype = {}
        elif isinstance(value, np.character):
            # numpy character scalar (rare)
            return value
        elif isinstance(value, np.number):
            return value
        else:
            default_dtype = {}


        if config.PIL_AVAILABLE and "PIL" in sys.modules:
            import PIL.Image

            if isinstance(value, PIL.Image.Image):
                return np.asarray(value, **self.np_array_kwargs)
        if config.TORCHVISION_AVAILABLE and "torchvision" in sys.modules:
            from torchvision.io import VideoReader

            if isinstance(value, VideoReader):
                return value  # TODO(QL): set output to np arrays ?
        if config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules:
            from torchcodec.decoders import AudioDecoder, VideoDecoder

            if isinstance(value, (VideoDecoder, AudioDecoder)):
                return value  # TODO(QL): set output to np arrays ?

        # Merge default dtype with provided kwargs; default_dtype may be empty.
        merged_kwargs = {**default_dtype, **self.np_array_kwargs}
        return np.asarray(value, **merged_kwargs)

    def _recursive_tensorize(self, data_struct):
        # support for torch, tf, jax etc.
        if config.TORCH_AVAILABLE and "torch" in sys.modules:
            import torch

            if isinstance(data_struct, torch.Tensor):
                return self._tensorize(data_struct.detach().cpu().numpy()[()])
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, (np.ndarray, np.character, np.number)):
            data_struct = data_struct.__array__()
        # support for nested types like struct of list of struct
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:
                # Use internal recursive helper to avoid extra map_nested overhead.
                return self._consolidate([self._recursive_tensorize(substruct) for substruct in data_struct])
        if isinstance(data_struct, (list, tuple)):
            return self._consolidate([self._recursive_tensorize(substruct) for substruct in data_struct])
        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict):
        return map_nested(self._recursive_tensorize, data_struct, map_list=False)

    def format_row(self, pa_table: pa.Table) -> Mapping:
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: pa.Table) -> np.ndarray:
        column = self.numpy_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        column = self.recursive_tensorize(column)
        column = self._consolidate(column)
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        batch = self.numpy_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        batch = self.recursive_tensorize(batch)
        for column_name in batch:
            batch[column_name] = self._consolidate(batch[column_name])
        return batch
