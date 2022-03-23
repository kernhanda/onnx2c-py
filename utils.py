from typing import List
import numpy as np
import onnx
import string

import tensor


def cify_name(name: str) -> str:
    forbidden = string.punctuation + string.whitespace
    return name.translate(str.maketrans(forbidden, '_' * len(forbidden)))


def parse_attribute_int(a: onnx.AttributeProto) -> int:
    if a.type != a.AttributeType.INT:
        raise ValueError("Not an int attribute")

    return a.i


def parse_attribute_ints(a: onnx.AttributeProto) -> List[int]:
    if a.type != a.AttributeType.INTS:
        raise ValueError("Not an ints attribute")

    return a.ints[:]


def parse_attribute_float(a: onnx.AttributeProto) -> float:
    if a.type != a.AttributeType.FLOAT:
        raise ValueError("Not a float attribute")

    return a.f


def parse_attribute_floats(a: onnx.AttributeProto) -> List[float]:
    if a.type != a.AttributeType.FLOATS:
        raise ValueError("Not a floats attribute")

    return a.floats[:]


def parse_attribute_string(a: onnx.AttributeProto) -> str:
    if a.type != a.AttributeType.STRING:
        raise ValueError("Not a string attribute")

    return a.s


def parse_attribute_strings(a: onnx.AttributeProto) -> str:
    if a.type != a.AttributeType.STRINGS:
        raise ValueError("Not a strings attribute")

    return a.strings


def parse_attribute_tensor(a: onnx.AttributeProto) -> tensor.Tensor:
    if a.type != a.AttributeType.TENSOR:
        raise ValueError("Not a tensor attribute")

    if not a.t:
        raise ValueError("No tensor in attribute")

    return tensor.parse_onnx_tensor(a.t)


def cast_to_ndim_arrayptr(t: tensor.Tensor, shortname: str) -> str:
    idxstr = "".join([f"[{d}]" for d in t.data.shape[1:]])

    # float (*X)[1][2]
    lhs = f"{t.data_type_str}(*{shortname}){idxstr}"

    # (float (*)[1][2])tensor_nodename_42
    rhs = f"({t.data_type_str} (*){idxstr}){t.cname}"

    res = f"{lhs} = {rhs};"

    return res


def is_splatted(t: tensor.Tensor, v) -> bool:
    if not t.isConst: return False

    return np.all(t.data == v)
