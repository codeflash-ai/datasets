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

# Lint as: python3
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from .. import config
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter


if TYPE_CHECKING:
    import tensorflow as tf


class TFFormatter(TensorFormatter[Mapping, "tf.Tensor", Mapping]):
    def __init__(self, features=None, token_per_repo_id=None, **tf_tensor_kwargs):
        super().__init__(features=features, token_per_repo_id=token_per_repo_id)
        self.tf_tensor_kwargs = tf_tensor_kwargs


        # Cached modules / classes to avoid repeated imports and lookups
        self._tf = None
        self._pil_image_cls = None
        self._video_reader_cls = None
        self._torchcodec_audio_video_classes = None
        self._torch_module = None

    def _consolidate(self, column):
        # match original behavior: import tf at top so ImportError behavior is preserved
        tf = self._get_tf()

        # Fast path checks and caching of first element attributes
        if column and isinstance(column, list):
            first = column[0]
            # check for identical tensors (shape & dtype)
            try:
                first_shape = first.shape
                first_dtype = first.dtype
            except Exception:
                first_shape = None
                first_dtype = None

            if first_shape is not None and first_dtype is not None:
                all_same = True
                for x in column:
                    if not (isinstance(x, tf.Tensor) and x.shape == first_shape and x.dtype == first_dtype):
                        all_same = False
                        break
                if all_same:
                    return tf.stack(column)

                # only rag 1-D tensors, otherwise some dimensions become ragged even though they were consolidated
                all_ragged_1d = True
                for x in column:
                    if not (
                        isinstance(x, (tf.Tensor, tf.RaggedTensor))
                        and getattr(x, "ndim", None) == 1
                        and getattr(x, "dtype", None) == first_dtype
                    ):
                        all_ragged_1d = False
                        break
                if all_ragged_1d:
                    return tf.ragged.stack(column)


        return column

    def _tensorize(self, value):
        # match original behavior: import tf at top so ImportError behavior is preserved
        tf = self._get_tf()


        if value is None:
            return value

        # determine default dtype only when applicable
        default_dtype = None  # use None to avoid small dict allocation when not needed

        if isinstance(value, (np.number, np.ndarray)):
            # safe to access .dtype for numpy scalar and ndarray
            try:
                if np.issubdtype(value.dtype, np.integer):
                    default_dtype = {"dtype": tf.int64}
                elif np.issubdtype(value.dtype, np.floating):
                    default_dtype = {"dtype": tf.float32}
            except Exception:
                default_dtype = None

        # PIL image handling: only if PIL module already loaded and available
        pil_cls = self._get_pil_image_cls()
        if pil_cls is not None:
            # import performed in _get_pil_image_cls to keep semantics
            if isinstance(value, pil_cls):
                value = np.asarray(value)

        # torchvision VideoReader handling: only if torchvision already loaded
        video_reader_cls = self._get_video_reader_cls()
        if video_reader_cls is not None:
            if isinstance(value, video_reader_cls):
                return value  # TODO(QL): set output to tf tensors ?

        # torchcodec handling: only if torchcodec already loaded
        tc_classes = self._get_torchcodec_classes()
        if tc_classes is not None:
            if isinstance(value, tc_classes):
                return value  # TODO(QL): set output to jax arrays ?

        # Avoid creating a new dict when not necessary
        if default_dtype is None:
            if self.tf_tensor_kwargs:
                return tf.convert_to_tensor(value, **self.tf_tensor_kwargs)
            else:
                return tf.convert_to_tensor(value)
        else:
            # need to merge default dtype with provided kwargs
            combined = dict(default_dtype)
            if self.tf_tensor_kwargs:
                combined.update(self.tf_tensor_kwargs)
            return tf.convert_to_tensor(value, **combined)

    def _recursive_tensorize(self, data_struct):
        # match original behavior: import tf at top so ImportError behavior is preserved
        tf = self._get_tf()

        # support for torch, tf, jax etc.
        torch_mod = self._get_torch()
        if torch_mod is not None:
            if isinstance(data_struct, torch_mod.Tensor):
                # preserve original conversion semantics
                return self._tensorize(data_struct.detach().cpu().numpy()[()])
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, tf.Tensor):
            data_struct = data_struct.__array__()
        # support for nested types like struct of list of struct
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:  # tf tensors cannot be instantied from an array of objects
                return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, (list, tuple)):
            return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict):
        return map_nested(self._recursive_tensorize, data_struct, map_list=False)

    def format_row(self, pa_table: pa.Table) -> Mapping:
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: pa.Table) -> "tf.Tensor":
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

    def _get_tf(self):
        if self._tf is None:
            import tensorflow as tf
            self._tf = tf
        return self._tf

    def _get_pil_image_cls(self):
        # Preserve original behavior: only import PIL.Image if PIL is already in sys.modules and config allows it
        if not (config.PIL_AVAILABLE and "PIL" in sys.modules):
            return None
        if self._pil_image_cls is None:
            import PIL.Image
            self._pil_image_cls = PIL.Image.Image
        return self._pil_image_cls

    def _get_video_reader_cls(self):
        if not (config.TORCHVISION_AVAILABLE and "torchvision" in sys.modules):
            return None
        if self._video_reader_cls is None:
            from torchvision.io import VideoReader
            self._video_reader_cls = VideoReader
        return self._video_reader_cls

    def _get_torchcodec_classes(self):
        if not (config.TORCHCODEC_AVAILABLE and "torchcodec" in sys.modules):
            return None
        if self._torchcodec_audio_video_classes is None:
            from torchcodec.decoders import AudioDecoder, VideoDecoder
            self._torchcodec_audio_video_classes = (VideoDecoder, AudioDecoder)
        return self._torchcodec_audio_video_classes

    def _get_torch(self):
        if not (config.TORCH_AVAILABLE and "torch" in sys.modules):
            return None
        if self._torch_module is None:
            import torch
            self._torch_module = torch
        return self._torch_module
