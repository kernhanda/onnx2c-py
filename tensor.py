import numpy as np
import onnx
import onnx.mapping
import onnx.numpy_helper as nph
from typing import Any

import utils


class Tensor:

    def __init__(
        self,
        name: str = "",
        doc: str = "",
        data: Any = None,
        raw_data: bytes = b'',
        generate=True,
        initialize=False,
        isConst=False,
        isIO=False,
        isRecursive=False
    ) -> None:
        """
        A entity that implements ONNX graph edges,
        i.e. the data buffers a ONNX node produces or consumes

        Args:
            generate (bool, optional): generate code (i.e global definition) for this Tensor. Defaults to True.
            initialize (bool, optional): generate initialization from data in data_buffer. Defaults to False.
            isConst (bool, optional): constant value. Defaults to False.
            isIO (bool, optional): is a parameter passed to the entry function of the graph. Defaults to False.
            isRecursive (bool, optional): tensor that one node uses both output and input. may additionally be used as input for other nodes. Defaults to False.
        """

        self.generate = generate
        self.initialize = initialize
        self.isConst = isConst
        self.isIO = isIO
        self.isRecursive = isRecursive

        self.name = name
        self.data = data
        self.raw_data = raw_data
        self.doc = doc

    @property
    def cname(self):
        return utils.cify_name(self.name)

    @property
    def data_elem_size(self):
        return self.data.itemsize

    @property
    def data_num_elem(self):
        return self.data.size

    @property
    def rank(self):
        return len(self.data.shape)

    @property
    def data_type_str(self):
        return {
            np.float32: "float",
            np.double: "double",
            np.int8: "int8_t",
            np.uint8: "uint8_t",
            np.int16: "int16_t",
            np.uint16: "uint16_t",
            np.int32: "int32_t",
            np.uint32: "uint32_t",
            np.int64: "int64_t",
            np.uint64: "uint64_t",
            np.dtype('float16'): "bfloat_t",  # native numpy does not support bfloat16
        }[self.data.dtype]

    @property
    def str_dimensions(self):
        return ' '.join(map(str, self.data.shape))

    @property
    def is_high_precision_numeric(self) -> bool:
        return self.data.dtype in [
            np.uint32,
            np.uint64,
            np.int32,
            np.int64,
            np.float16,
            np.float32,
            np.float64
        ]

    @property
    def is_all_fp(self) -> bool:
        ...

    @property
    def is_non_bfloat(self) -> bool:
        ...

    @property
    def is_int64(self)->bool:
        ...

    @property
    def is_8bit(self)->bool:
        ...

    @property
    def is_any_int(self)->bool:
        ...

    @property
    def is_unsigned_int(self)->bool:
        ...

    @property
    def is_signed_int(self)->bool:
        ...



def parse_onnx_tensor(t: onnx.TensorProto):
    T = onnx.TensorProto

    generate = True
    initialize = True
    isIO = False
    isConst = True

    if t.data_location != T.DataLocation.DEFAULT:
        raise RuntimeError(f"unhandled: non-default data location in t {t.name}")
    if t.segment:
        raise RuntimeError(f"unhandled: segmented data in t {t.name}")
    if t.data_type == T.UNDEFINED:
        raise RuntimeError(f"unknown data type in t {t.name}")

    data = nph.to_array(t)
    raw_data = t.raw_data
    name = t.name
    doc = t.doc_string

    return Tensor(
        generate=generate,
        initialize=initialize,
        isIO=isIO,
        isConst=isConst,
        data=data,
        raw_data=raw_data,
        name=name,
        doc=doc
    )


def parse_onnx_value_info(vi: onnx.ValueInfoProto) -> Tensor:
    if not vi.type.tensor_type:
        raise ValueError("Cannot handle non-tensor value info")

    tt = vi.type.tensor_type
    shape: onnx.TensorShapeProto = tt.shape

    array_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tt.elem_type]
    array_shape = [d.dim_value for d in shape.dim]

    initialize = False
    generate = False
    isIO = True
    isConst = False
    name = vi.name
    doc = vi.doc_string
    data = np.ndarray(shape=array_shape, dtype=array_type)

    return Tensor(name=name, doc=doc, data=data, generate=generate, initialize=initialize, isIO=isIO, isConst=isConst)
