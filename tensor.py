import numpy as np
import onnx
import onnx.mapping
import onnx.numpy_helper as nph
from typing import Any, TextIO

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
            np.bool: "bool",
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
        }[self.data.dtype]

    @property
    def str_dimensions(self):
        return ' '.join(map(str, self.data.shape))

    @property
    def is_high_precision_numeric(self) -> bool:
        # TODO: support fp16
        # TODO: Support bfloat16
        return self.data.dtype in [np.uint32, np.uint64, np.int32, np.int64, np.float16, np.float32, np.float64]

    @property
    def is_all_fp(self) -> bool:
        # TODO: support fp16
        return self.data.dtype in [np.float32, np.float64]

    @property
    def is_non_bfloat(self) -> bool:
        # TODO: Support bfloat16
        return True

    @property
    def is_int64(self) -> bool:
        return self.data.dtype == np.int64

    @property
    def is_8bit(self) -> bool:
        return self.data.dtype in [np.int8, np.uint8]

    @property
    def is_any_int(self) -> bool:
        return self.is_unsigned_int() or self.is_signed_int()

    @property
    def is_unsigned_int(self) -> bool:
        return self.data.dtype in [
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]

    @property
    def is_signed_int(self) -> bool:
        return self.data.dtype in [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ]

    @property
    def is_used(self) -> bool:
        return bool(self.name)

    def print_element(self, destination: IO, element: int):
        destination.write(str(self.data[element]).encode('utf-8'))
        if self.is_all_fp():
            destination.write(b"f")

    def print_tensor_initializer(self, destination: IO):
        self._print_tensor_initializer(destination, dim=0, offs=0)

    def _print_tensor_initializer(self, destination: TextIO, dim: int = 0, offs: int = 0):
        """
            Print the tensor initialization values.
            dim: the dimension from which to print.
            offs: the offset into this dimension from where to print.
            This function recurses back into itself to print all more inner dimenstions there are.
            I.e. if calling with dim=0, offs=0 (which are default values),
            it prints the entire variable initialzation.
        """
        if self.data.shape[dim] == 0:
            return

        destination.write("  " * dim)
        destination.write("{")

        if dim < (len(self.data.shape) - 1):
            destination.write("\n")

            for i in range(self.data.shape[dim]):
                remaining_dims = 1

                for j in self.data.shape[dim + 1:]:
                    remaining_dims *= j

                self.print_tensor_initializer(destination, dim + 1, offs + i * remaining_dims)

                if i < (self.data.shape[dim] - 1):
                    destination.write(",")

                destination.write("\n")

            destination.write("  " * dim)

        else:
            for i in range(self.data.shape[dim]):
                element = offs + i

                # TODO: It might be preferable to inline this function
                self.print_element(destination, element)

                if i < (self.data.shape[dim] - 1):
                    destination.write(", ")

        destination.write("}")

    def print_tensor(self, destination: TextIO = None, alternate_name="", callsite=False, const=False):
        res: str = ""

        if not callsite:
            if self.isConst or const:
                res += "const "

            res += self.data_type_str() + " "

        res += alternate_name or self.cname

        if not callsite:
            res += ''.join([f"[{i}]" for i in self.data.shape])

        if destination:
            destination.write(res)
        else:
            return res

    def print_tensor_callsite(self) -> str:
        return self.print_tensor(alternate_name="", callsite=True, const=False)

    def print_tensor_as_const(self, destination: TextIO = None, alternate_name: str = "", callsite=False) -> str:
        return self.print_tensor(destination=destination, alternate_name=alternate_name, callsite=callsite, const=True)


def parse_onnx_tensor(t: onnx.TensorProto):
    T = onnx.TensorProto

    if t.data_location != T.DataLocation.DEFAULT:
        raise RuntimeError(f"unhandled: non-default data location in tensor {t.name}")
    if t.segment:
        raise RuntimeError(f"unhandled: segmented data in tensor {t.name}")
    if t.data_type == T.UNDEFINED:
        raise RuntimeError(f"unknown data type in tensor {t.name}")
    if t.data_type == T.BFLOAT16:
        raise RuntimeError(f"unhandled: bfloat16 data type in tensor {t.name}")
    if t.data_type == T.FLOAT16:
        raise RuntimeError(f"unhandled: bfloat16 data type in tensor {t.name}")

    generate = True
    initialize = True
    isIO = False
    isConst = True

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

    T = onnx.TensorProto
    tt = vi.type.tensor_type
    shape: onnx.TensorShapeProto = tt.shape

    if tt.elem_type == T.BFLOAT16:
        raise RuntimeError(f"unhandled: bfloat16 data type in tensor {vi.name}")
    if tt.elem_type == T.FLOAT16:
        raise RuntimeError(f"unhandled: bfloat16 data type in tensor {vi.name}")

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
