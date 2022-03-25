import logging
import numpy as np
from abc import ABC, abstractmethod
from functools import reduce
from math import ceil, floor, fmod
from typing import TextIO, List, Tuple

import onnx

import tensor
import utils

_anonymous_node_counter = 0

FuncParam = Tuple[tensor.Tensor, str]


class Node(ABC):
    onnx_ir_version = 0

    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        self.is_resolved = is_resolved
        self.onnx_node = onnx_node
        self.onnx_name: str = onnx_name or (onnx_node.name if onnx_node else "")
        self.op_name: str = op_name or (onnx_node.op_type if onnx_node else "")

        self._input_params: List[FuncParam] = []
        self._output_params: List[FuncParam] = []

        if not self.onnx_name:
            # anonymous nodes
            global _anonymous_node_counter
            self.onnx_name = f"anonymous_{self.op_name}_{_anonymous_node_counter}"
            _anonymous_node_counter += 1

    def parse_attributes(self):
        if self.onnx_node.attribute:
            raise RuntimeError(f"Attribute parsing not implemented for node operation type {self.onnx_node.op_type}")

    @abstractmethod
    def print(self, destination: TextIO):
        ...

    @abstractmethod
    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        ...

    def print_func_params(self, destination: TextIO, decorate: bool):
        params: List[str] = []

        if decorate:
            params += [t.print_tensor_as_const(alternate_name=name) for t, name in self._input_params]

            params += [t.print_tensor(alternate_name=name) for t, name in self._output_params if t.is_used]
        else:
            params += [t.print_tensor_callsite() for t, _ in self._input_params]

            params += [t.print_tensor_callsite() for t, _ in self._output_params if t.is_used]

        destination.write(", ".join(params))

    def print_func_params_shapes(self, destination: TextIO):
        self.print_func_params(destination=destination, decorate=True)

    def print_func_params_callsite(self, destination: TextIO):
        self.print_func_params(destination=destination, decorate=False)

    def is_output_used(self, index: int) -> bool:
        # ONNX spec:
        # "There are two ways to leave an optional input or output unspecified:
        # the first, available only for trailing inputs and outputs, is to simply
        # not provide that input; the second method is to use an empty string in
        # place of an input or output name."
        if index >= len(self.onnx_node.output):
            return False

        # If it has a name, it's used
        return self.onnx_node.output[index] != ""

    @property
    def c_name(self):
        return "node_" + utils.cify_name(self.onnx_name)

    def multidirectional_broadcast_size(self, A: List[int], B: List[int]) -> List[int]:
        diff_len = len(A) - len(B)
        if diff_len > 0:
            B = [1] * diff_len + list(B)
        elif diff_len < 0:
            A = [1] * (diff_len * -1) + list(A)

        res = []
        for a, b in zip(A, B):
            if a == 1 or b == 1:
                res.append(max(a, b))
            elif a == b:
                res.append(a)
            else:
                raise RuntimeError(f"multidirectional_broadcast: bad tensor shapes for node {self.onnx_name}")

        return res

    def _register_input(self, t: tensor.Tensor, name: str):
        self._input_params.append((t, name))

    def _register_output(self, t: tensor.Tensor, name: str):
        self._output_params.append((t, name))


class Elementwise_2(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        # input and output: C = A ? B
        self.A: tensor.Tensor = None
        self.B: tensor.Tensor = None
        self.C: tensor.Tensor = None
        self.output_is_bool = False

        # Union of attributes over implemented nodes
        self.shift_dir = "NOT_GIVEN"    # mandatory for BitShift but no default
        self.fmod: int = 0

        if self.op_name == "Add":
            self.operation = lambda a, b: a + "+" + b + ";"
        elif self.op_name == "And":
            self.operation = lambda a, b: a + "&" + b + ";"
        elif self.op_name == "BitShift":
            self.operation = lambda a, b: a + (">>" if self.shift_dir == "RIGHT" else "<<") + b + ";"
        elif self.op_name == "Div":
            self.operation = lambda a, b: a + "/" + b + ";"
        elif self.op_name == "Equal":
            self.output_is_bool = True

            # From onnx2c
            # NB: specs don't define what kind of equality is meant when inputs are floating point
            # This passes currently existing ONNX unit tests...
            self.operation = lambda a, b: a + "==" + b + ";"
        elif self.op_name == "Greater":
            self.output_is_bool = True
            self.operation = lambda a, b: a + ">" + b + ";"
        elif self.op_name == "GreaterOrEqual":
            self.output_is_bool = True
            self.operation = lambda a, b: a + ">=" + b + ";"
        elif self.op_name == "Less":
            self.output_is_bool = True
            self.operation = lambda a, b: a + "<" + b + ";"
        elif self.op_name == "LessOrEqual":
            self.output_is_bool = True
            self.operation = lambda a, b: a + "<=" + b + ";"
        elif self.op_name == "Mod":

            def do_mod(a, b):
                if self.fmod:
                    return "fmod(" + a + ", " + b + ");"

                raise ValueError("Non fmod Mod operator definition is not clear in ONNX specification")

            self.operation = do_mod
        elif self.op_name == "Mul":
            self.operation = lambda a, b: a + "*" + b + ";"
        elif self.op_name == "Or":
            self.output_is_bool = True
            # TODO: bitwise AND is used above, logical OR is used here
            self.operation = lambda a, b: a + "||" + b + ";"
        elif self.op_name == "Pow":
            # TODO: don't use powf for integers
            self.operation = lambda a, b: f"powf({a}, {b});"
        elif self.op_name == "PRelu":
            self.operation = lambda a, b: f"{a} < 0 ? {a} * {b} : {a};"
        elif self.op_name == "Xor":
            self.output_is_bool = True
            self.operation = lambda a, b: a + "^" + b + ";"
        elif self.op_name == "Sub":
            self.operation = lambda a, b: a + "-" + b + ";"
        else:
            raise ValueError(f"Elementwise_2 operand {self.op_name} not implemented")

    def parse_attributes(self):
        for a in self.onnx_node.attribute:
            logging.debug(f"Parsing attribute {a.name}")

            if a.name == "direction":
                self.shift_dir = utils.parse_attribute_string(a)

            elif a.name == "fmod":
                self.fmod = utils.parse_attribute_int(a)

            else:
                raise ValueError("unknown attribute")

    def print(self, destination: TextIO):
        destination.write(
            f"\t/* {self.op_name}\n"
            f"\t   Implemented with Elementwise_2 template.\n"
            f"\t   Attributes (these are the union of attributes for all 2-element-wise\n"
            f"\t               operands. So most likely these values are ignored by onnx2c).\n"
            f"\t   shift_dir: {self.shift_dir}\n"
            f"\t   fmod: {self.fmod}\n"
            f"\t */\n"
        )

        A = self.A
        B = self.B
        C = self.C

        # if either A or B does not have enough dimensions, prepend
        # dimensions of 1 to match rank of C
        pad_A = [0] * (C.rank - (A.rank or A.size)) + (A.shape or [A.size])
        pad_B = [0] * (C.rank - (B.rank or B.size)) + (B.shape or [B.size])

        # print out the loops over all C dimensions.
        # at the same time, create the indexing strings into A and B
        Aidx = "A"
        Bidx = "B"
        Cidx = "C"
        for r in range(C.rank):
            lv = f"i{r}"

            destination.write(f"\tfor (unsigned {lv} = 0; {lv} < {C.shape[r]}; ++{lv}) {{\n")

            if pad_A[r] == 1:
                Aidx += "[0]"
            elif pad_A[r] != 0:
                Aidx += f"[{lv}]"

            if pad_B[r] == 1:
                Bidx += "[0]"
            elif pad_B[r] != 0:
                Bidx += f"[{lv}]"

            Cidx += f"[{lv}]"

        destination.write(f"\t\t{Cidx} = {self.operation(Aidx, Bidx)};\n")

        destination.write("\t}\n" * C.rank)

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        A = self.A = inputs[0]
        B = self.B = inputs[1]

        res_shape = self.multidirectional_broadcast_size(A.shape, B.shape)

        t = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=np.bool if self.output_is_bool else A.data.dtype))
        self.C = t

        self._register_input(A, "A")
        self._register_input(B, "B")
        self._register_output(self.C, "C")

        return [self.C]


class SpatialFilter(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.x: tensor.Tensor = None
        self.w: tensor.Tensor = None
        self.y: tensor.Tensor = None

        self.auto_pad = "NOTSET"
        self.dilations: List[int] = []
        self.group: int = None
        self.kernel_shape: List[int] = []
        self.pads: List[int] = []
        self.strides: List[int] = []

    def parse_attributes(self):
        for a in self.onnx_node.attribute:
            if a.name == "auto_pad":
                self.auto_pad = utils.parse_attribute_string(a)
            elif a.name == "dilations":
                self.dilations = utils.parse_attribute_ints(a)
            elif a.name == "group":
                self.group = utils.parse_attribute_int(a)
            elif a.name == "kernel_shape":
                self.kernel_shape = utils.parse_attribute_ints(a)
            elif a.name == "pads":
                self.pads = utils.parse_attribute_ints(a)
            elif a.name == "strides":
                self.strides = utils.parse_attribute_ints(a)

    def resolve_strides(self):
        if not self.strides:
            self.strides = [1] * (self.x.rank - 2)

    def resolve_kernel_shape(self):
        if not self.kernel_shape:
            self.kernel_shape = self.w.shape[2:]

    def resolve_dilations(self):
        if not self.dilations:
            self.dilations = [1] * (self.x.rank - 2)

    def resolve_pads(self):
        if not self.pads:
            num_data_dim = self.x.rank - 2
            self.pads = [0] * (num_data_dim * 2)

            for i in range(num_data_dim):
                if self.auto_pad in ["VALID", "NOTSET"]:
                    self.pads[i] = self.pads[i + num_data_dim] = 0
                else:
                    # TODO: dilations and strides might cause need for bigger paddings
                    # TODO: handle case where uneven padding is needed
                    self.pads[i] = self.pads[i + num_data_dim] = self.kernel_shape[i] // 2

    def resolve_output_size(self) -> List[int]:
        num_data_dim = self.x.rank - 2
        res: List[int] = [
            self.x.shape[0],    # batch size
            self.w.shape[0],    # number of feature maps
        ]

        # From onnx2c
        # Not sure if the naming is correct. Here
        # kernel: the (number of) weights of the filter
        # filter: the spatial placement of the kernel weights
        # NB: 'dilation==1' is what is used for "no spacing in the filter"
        filter_size = list(map(lambda k, d: k + (k - 1) * (d - 1), self.kernel_shape, self.dilations))

        for xdim in range(2, self.x.rank):
            dim = xdim - 2

            # From ONNX Operators.md:
            # SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input.
            # "match" here means "is equal".
            if self.auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
                outdim = self.x.shape[xdim]
            elif self.auto_pad in ["NOTSET", "VALID"]:
                # padded input
                input_size = self.x.shape[xdim] + self.pads[dim] + self.pads[dim + num_data_dim]

                # From onnx2c
                # [ 0 1 2 3 4 5 6 7 8 9  ]
                #                |kern=3|
                # last output=7
                last_out = input_size - filter_size[dim]
                outdim = last_out // self.strides[dim] + 1

            res.append(outdim)

        return res

    def print_header_info_comment(self, destination: TextIO):
        def lstr(l: List):
            return ' '.join(map(str, l))

        destination.write(
            f"\t/* {self.op_name}\n"
            f"\t *\n"
            f"\t * auto_pad: {self.auto_pad}\n"
            f"\t * dilations: {lstr(self.dilations)}\n"
            f"\t * group: {self.group}\n"
            f"\t * kernel_shape: {lstr(self.kernel_shape)}\n"
            f"\t * pads: {lstr(self.pads)}\n"
            f"\t * strides: {lstr(self.strides)}\n"
            f"\t */\n"
        )

    # From onnx2c
    # Print the loops of the convolution.
    # This version has checks in the innermost loop for checking when
    # the kernel hits paddings.
    # This (probably, unless compliers are getting *real* amazing) causes
    # a lot of overhead. A.k.a. optmization opportunities.
    #
    # Three callbacks to pure virtual functions are used:
    # - to initialize output cell
    # - to calculate input cell / kernel cell (this is the calculation in the innermost loop)
    # - to finalize the output cell
    @abstractmethod
    def print_output_cell_init(self, destination: TextIO, y_idx: str = ""):
        ...

    @abstractmethod
    def print_output_cell_calc(self, destination: TextIO, x_idx: str = "", w_idx: str = "", y_idx: str = ""):
        ...

    @abstractmethod
    def print_output_cell_finalize(self, destination: TextIO, y_idx: str = ""):
        ...

    def print_loop_with_padding_checks(self, destination: TextIO):
        x = self.x
        y = self.y
        w = self.w
        group = self.group
        pads = self.pads
        kernel_shape = self.kernel_shape
        dilations = self.dilations
        strides = self.strides

        n_data_dims = x.rank - 2
        batch_size = x.shape[0]
        channels = x.shape[1]

        maps = w.shape[0] if w else 0

        x_idx = reduce(lambda x, y: x + y, (f"[i{i}]" for i in range(n_data_dims)), "[b][c]")
        in_kern_idxs = reduce(lambda x, y: x + y, (f"[ii{i}]" for i in range(n_data_dims)), "[b][c]")
        y_idx = reduce(lambda x, y: x + y, (f"[o{i}]" for i in range(n_data_dims)), "[b][c]")

        destination.write(f"\tfor (uint64_t b = 0; b < {batch_size}; ++b) {{\n")

        if w and group == 1:
            destination.write(f"\tfor (uint64_t m = 0; m < {maps}; ++m) {{\n")

        elif w and group > 1:
            destination.write(
                f"\tuint64_t go = {maps//group}; // output group size, i.e. maps/group\n"
                f"\tuint64_t gi = {channels//group}; // inout group size, i.e. channels/group\n"
                f"\tfor (uint64_t g = 0; g < {group}; ++g) {{\n"
                f"\tfor (uint64_t m = go * g; m < go * (g + 1); ++m) {{\n"
            )

        else:
            destination.write(f"\tfor (uint64_t c = 0; c < {channels}; ++c) {{\n")

        for i in range(n_data_dims):
            o_idx = f"o{i}"
            i_idx = f"i{i}"

            destination.write(
                f"\t\tfor (int32_t {o_idx} = 0, {i_idx} = {-pads[i]}; "
                f"{o_idx} < {y.shape[2 + i]}; "
                f"++{o_idx}, {i_idx} += {strides[i]}) {{\n"
            )

        self.print_output_cell_init(destination=destination, y_idx=y_idx)

        if w and group == 1:
            destination.write(f"\t\t\tfor (int32_t c = 0; c < {channels}; ++c) {{\n")

        elif w and group > 1:
            destination.write(f"\t\t\tfor (int32_t c = gi * g; c < gi * (g + 1); ++c) {{\n")

        # loop over channels and kernel
        for i in range(n_data_dims):
            idx = f"k{i}"
            destination.write(f"\t\t\tfor (uint32_t {idx} = 0; {idx} < {kernel_shape[i]}; ++{idx}) {{\n")

        for i in range(n_data_dims):
            destination.write(
                f"\t\t\t\tint ii{i} = i{i} + k{i} * {dilations[i]};\n"
                f"\t\t\t\tif (ii{i} < 0) continue;\n"
                f"\t\t\t\tif (ii{i} >= {x.shape[2 + i]}) continue;\n"
            )

        self.print_output_cell_calc(destination=destination, x_idx=in_kern_idxs, w_idx="", y_idx=y_idx)

        destination.write("\t\t\t} /* k */\n" * n_data_dims)

        if w:
            destination.write("\t\t\t} /* c */\n")

        self.print_output_cell_finalize(destination=destination, y_idx=y_idx)

        destination.write("\t\t} /* o */\n" * n_data_dims)

        destination.write("\t} /* m or c, depending on this node's operator */\n")

        if w and group > 1:
            destination.write("\t} /* g */\n")

        destination.write("\t} /* b */\n")


class Pooling(SpatialFilter):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.ceil_mode = 0
        self.count_include_pad = 0
        self.storage_order = 0

    def parse_attributes(self):
        super().parse_attributes()

        for a in self.onnx_node.attribute:
            if a.name == "ceil_mode":
                self.ceil_mode = utils.parse_attribute_int(a)

            elif a.name == "count_include_pad":
                self.count_include_pad = utils.parse_attribute_int(a)

            elif a.name == "storage_order":
                self.storage_order = utils.parse_attribute_int(a)

    def resolve_output_size(self) -> List[int]:
        x = self.x
        pads = self.pads
        kernel_shape = self.kernel_shape
        dilations = self.dilations
        strides = self.strides
        auto_pad = self.auto_pad
        ceil_mode = self.ceil_mode
        count_include_pad = self.count_include_pad
        storage_order = self.storage_order

        res = list(x.shape[:2])

        data_dims = x.rank - 2
        pad_shapes = [pads[i] + pads[i + data_dims] for i in range(data_dims)]

        for i in range(2, x.rank):
            in_dim = x.shape[i]
            kernel = kernel_shape[i - 2]
            dilation = dilations[i - 2] if dilations else 1
            stride = strides[i - 2]

            if auto_pad == "NOTSET":
                pad_sh = pad_shapes[i - 2]

                if ceil_mode:
                    d = ceil((in_dim + pad_sh - ((kernel - 1) * dilation + 1)) / stride + 1)

                else:
                    d = floor((in_dim + pad_sh - ((kernel - 1) * dilation + 1)) / stride + 1)

            elif auto_pad == "VALID":
                d = ceil((in_dim - ((kernel - 1) * dilation + 1) + 1) / stride)

            else:    # "SAME_UPPER", "SAME_LOWER"
                d = ceil(in_dim / stride)

            res.append(d)

        return res

    # From onnx2c
    # The auto_pad mess:
    # pads are needed to calculate output shape, but output shape is needed to calculate pads
    # Run this after resolve_output_size() to patch up
    def update_pads(self):
        auto_pad = self.auto_pad
        x = self.x
        y = self.y
        strides = self.strides
        kernel_shape = self.kernel_shape
        dilations = self.dilations
        pads = self.pads

        if auto_pad in ["NOTSET", "VALID"]:
            return

        data_dims = x.rank - 2

        # Calculate pads for the SAME_* cases that need the output shape
        for i in range(data_dims):

            # From onnx2c
            # NB: onnx2c architecture does not allow figuring out the output shape at this stage
            # (especially since the onnx spec says it is a function of input, strides, pads &c).
            # The auto_pad attribute for AveragePool is deprecated anyway. Probably just for this confusion.
            # This tries to be some sort of band-aid: assume the output size is the same as input size
            # which is the usual(?) reason to use "same" padding on the network design level.

            input_size = x.shape[i + 2]
            output_size = y.shape[i + 2]
            pad_shape = (output_size - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - input_size
            pads[i] = pads[i + data_dims] = pad_shape // 2

            if pad_shape & 1:

                if auto_pad == "SAME_LOWER":
                    pads[i] += 1

                else:
                    pads[i + data_dims] += 1


class MaxPool(Pooling):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.indices: tensor.Tensor = None
        self.pad_shapes: List[int] = []

    def parse_attributes(self):
        super().parse_attributes()

        for a in self.onnx_node.attribute:
            if a.name == "storage_order":
                raise ValueError("Unimplemented: MaxPool storage_order attribute")

    def print_output_cell_init(self, destination: TextIO, y_idx: str = ""):
        x = self.x
        dtype = x.data_type_str

        type_min_value = {
            "float": "-FLT_MAX",
            "int8_t": "INT8_MIN",
            "uint8_t": "0",
            "int32_t": "INT32_MIN"
        }.get(dtype)

        if not type_min_value:
            raise ValueError("Unimplemented: minimum value for this type")

        destination.write(f"\t\t\t{dtype} curmax = {type_min_value};\n")

        if self.indices.name:
            destination.write("\t\t\tint64_t curmaxind = -1;\n")

    def print_output_cell_calc(self, destination: TextIO, x_idx: str = "", w_idx: str = "", y_idx: str = ""):
        x = self.x
        indices = self.indices

        n_data_dims = x.rank - 2

        # Calculate how much one index means in terms of the Indices output.
        # Generate helper string for the next step.
        size_of_dim = [0] * x.rank
        size_of_dim[-1] = 1

        for i in reversed(range(n_data_dims)):
            size_of_dim[i] = size_of_dim[i + 1] * x.shape[i]

        indices_value = f"(b * {size_of_dim[0]}) + (c * {size_of_dim[1]})"
        for i in range(n_data_dims):
            indices_value += f" + (ii{i} * {size_of_dim[i + 2]})"

        destination.write(
            f"\t\t\t\tif (curmax < {x.cname}{x_idx}) {{\n"
            f"\t\t\t\tcurmax = MAX( curmax, {x.cname}{x_idx} );\n"
        )

        if self.indices.name:
            destination.write(f"\t\t\t\tcurmaxind = {indices_value};\n")

        destination.write("}\n")

    def print_output_cell_finalize(self, destination: TextIO, y_idx: str = ""):
        y = self.y
        indices = self.indices

        destination.write(f"\t\t\t{y.cname}{y_idx} = curmax;\n")

        if indices.name:
            destination.write(f"\t\t\t{indices.cname}{y_idx} = curmaxind;\n")

    def print(self, destination: TextIO):
        self.print_header_info_comment(destination=destination)
        self.print_loop_with_padding_checks(destination=destination)

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        self.x = inputs[0]

        self.resolve_strides()
        self.resolve_dilations()
        self.resolve_pads()
        self.resolve_kernel_shape()

        if self.storage_order:
            raise ValueError("Unimplemented: column-major storage_order")

        res_shape = self.resolve_output_size()
        res = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=self.x.data.dtype))

        self.y = res

        outputs = [self.y]

        self.update_pads()

        indices_out = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=np.int64))
        self.indices = indices_out

        outputs.append(indices_out)

        self._register_input(self.x, "x")
        self._register_output(self.y, "y")

        self._register_output(self.indices, "indices")

        return outputs


class AveragePool(Pooling):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.indices: tensor.Tensor = None

    def print_output_cell_init(self, destination: TextIO, y_idx: str = ""):
        destination.write(f"\t\t\t{self.y.data_type_str} curavg = 0;\n\t\t\tint numavg = 0;\n")

    def print_output_cell_calc(self, destination: TextIO, x_idx: str = "", w_idx: str = "", y_idx: str = ""):
        # Sum up the cells
        destination.write(f"\t\t\t\tnumavg += 1;\n\t\t\t\tcuravg += {self.x.cname}{x_idx};\n")

    def print_output_cell_finalize(self, destination: TextIO, y_idx: str = ""):
        if self.count_include_pad:
            numavg = reduce(lambda x, y: x * y, self.kernel_shape)

            destination.write(
                f"\t\t\t/* Counting padding into the average is requested */\n"
                f"\t\t\tnumavg = {numavg};\n"
            )

        destination.write(f"\t\t\t{self.y.cname}{y_idx} = curavg/numavg;\n")

    def print(self, destination: TextIO):
        self.print_header_info_comment(destination=destination)
        self.print_loop_with_padding_checks(destination=destination)

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        self.x = inputs[0]

        if not (self.x.is_all_fp or self.x.is_8bit):
            raise ValueError("Incorrect input for node")

        self.resolve_strides()
        self.resolve_dilations()
        self.resolve_pads()
        self.resolve_kernel_shape()

        res_shape = self.resolve_output_size()
        res = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=self.x.data.dtype))
        self.y = res

        outputs = [res]

        self.update_pads()

        # Optional indices vector
        indices_out = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=np.int64))
        self.indices = indices_out

        outputs.append(indices_out)

        self._register_input(self.x, "x")
        self._register_output(self.y, "y")
        self._register_output(self.indices, "indices")

        return outputs


class GlobalAveragePool(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.x: tensor.Tensor = None
        self.y: tensor.Tensor = None

    def print(self, destination: TextIO):
        x = self.x
        y = self.y

        batch_size = x.shape[0]
        num_channels = x.shape[1]

        destination.write(
            f"\t/* GlobalAveragePool */\n"
            f"\tfor (int32_t b = 0; b < {batch_size}; ++b) {{\n"
            f"\tfor (int32_t c; c < {num_channels}; ++c) {{\n"

        # From onnx2c
        # TODO: float16, double? accuracy vs speed...
            "\t\tfloat dimsum = 0.0f;\n"
        )

        dim_num_elem = 1    # number of elements averaged over
        in_idx_str = "x" + "[b][c]"    # start of the input element access string
        out_idx_str = "y" + "[b][c]"    # start of the output element access string

        for dim, dim_size in enumerate(x.shape[2:], 2):
            dim_num_elem *= dim_size

            dim_var = f"d{dim - 2}"
            in_idx_str += f"[{dim_var}]"
            out_idx_str += "[0]"

            destination.write(f"\t\tfor (int32_t {dim_var} = 0; {dim_var} < {dim}; ++{dim_var}) {{\n")

        destination.write(f"\t\t\tdimsum += {in_idx_str};\n")

        destination.write("\t\t}\n" * (x.rank - 2))

        destination.write(
            f"\t\t{out_idx_str} = dimsum / {dim_num_elem};\n"

        # close loop over b and c
            "\t}\n"
            "\t}\n"
        )

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        self.x = inputs[0]

        if not self.x.is_all_fp:
            raise ValueError("Incorrect input for node")

        res_shape = self.x.shape[:2] + [1] * (self.x.rank - 2)
        self.y = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=self.x.data.dtype))

        self._register_input(self.x, "x")
        self._register_output(self.y, "y")

        return [self.y]


class MatMul(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.A: tensor.Tensor = None
        self.B: tensor.Tensor = None
        self.Y: tensor.Tensor = None

    def print(self, destination: TextIO):
        A = self.A
        B = self.B
        Y = self.Y

        dtype = A.data_type_str

        if A.rank != 2:
            raise ValueError("Unimplemented: higher than 2D MatMul")

        rows = A.shape[0]
        cols = B.shape[1]
        inner = A.shape[1]
        inner2 = B.shape[0]
        if inner == 0:
            inner = 1

        # From onnx2c
        # TODO: handle the case of [N] * [Nx1] multiplication,
        #       i.e. shift rows to inner, set rows as 1
        #       and similarly, the case of input[1] being a 1D vector
        if inner != inner2:
            raise ValueError("MatMul input's inner dimensions don't match")

        destination.write(
            f"\t/* MatMul */\n"
            f"\tfor (uint32_t r = 0; r < {rows}; ++r)\n"
            f"\t\tfor (uint32_t c = 0; c < {cols}; ++c) {{\n"
            f"\t\t\tY[r][c] = 0;\n"
            f"\t\t\tfor (uint32_t i = 0; i < {inner}; ++i);\n"
            f"\t\t\t\tY[r][c] += A[r][i] * B[i][c];\n"
            f"\t\t}}\n"
        )

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        self.A = inputs[0]
        self.B = inputs[1]
        self._register_input(self.A, "A")
        self._register_input(self.B, "B")

        if not self.A.is_high_precision_numeric or not self.B.is_high_precision_numeric:
            raise ValueError("Incorrect input for MatMul")

        res_shape = self.result_dim(inputs)

        res = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=self.A.data.dtype))

        self.Y = res

        self._register_output(self.Y, "Y")

        return [self.Y]

    def result_dim(self, inputs: List[tensor.Tensor]) -> Tuple[int, int]:
        # TODO: this is the check for vectors. Check equivalent for N-dimensions: N > 2
        if inputs[0].shape[1] != 0 and inputs[1].shape[1] != 0:
            rows = inputs[0].shape[0]
            cols = inputs[1].shape[1]

        elif inputs[0].shape[1] == 0 and inputs[1].shape[1] == 0:
            raise ValueError("Bad input/unhandled: 2 vectors to MatMul")

        elif inputs[0].shape[1] == 0:
            cols = inputs[1].shape[1]

            if inputs[0].shape[0] == inputs[1].shape[0]:
                rows = 1
            else:
                rows = inputs[0].shape[0]

        else:
            rows = inputs[0].shape[0]

            if inputs[0].shape[1] == inputs[1].shape[0]:
                cols = 1
            else:
                cols = inputs[1].shape[0]

        return (rows, cols)


class Slice(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        # input and output
        self.output: tensor.Tensor = None
        self.data: tensor.Tensor = None
        self.starts: tensor.Tensor = None
        self.ends: tensor.Tensor = None

        # optional inputs
        self.axes: tensor.Tensor = None
        self.steps: tensor.Tensor = None

        # contents of the input tensors, attributes or default values; padded
        # to output dimensions in resolve_node().
        self.sta: List[int] = []
        self.en: List[int] = []
        self.ax: List[int] = []
        self.stp: List[int] = []

    def parse_attributes(self):
        for a in self.onnx_node.attribute:
            logging.debug(f"Parsing attribute {a.name}")

            if a.name == "axes":
                self.ax = utils.parse_attribute_ints(a)

            elif a.name == "starts":
                self.sta = utils.parse_attribute_ints(a)

            elif a.name == "ends":
                self.en = utils.parse_attribute_ints(a)

            else:
                raise ValueError("Unknown attribute to slice")

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        data = self.data = inputs[0]
        self._register_input(data, "data")

        if len(inputs) > 1:
            self.starts = inputs[1]
            self._register_input(self.starts, "starts")

        if len(inputs) > 2:
            self.ends = inputs[2]
            self._register_input(self.ends, "ends")

        if len(inputs) > 3:
            self.axes = inputs[3]
            self._register_input(self.axes, "axes")

        if len(inputs) > 4:
            self.steps = inputs[4]
            self._register_input(self.steps, "steps")

        starts = self.starts
        ends = self.ends
        axes = self.axes
        steps = self.steps

        if starts and not starts.isConst:
            raise ValueError("Non-const inputs to Slice not handled")

        if ends and not ends.isConst:
            raise ValueError("Non-const inputs to Slice not handled")

        # Set defaults. Override later if required
        sta = [0] * data.rank
        en = data.shape
        ax = list(range(data.rank))
        stp = [1] * data.rank

        # if axes are not provided as input, the rest of the limits must be provided in full
        # or we can't know which axes a limit applies to
        if not axes:
            expected_size = data.rank

        else:
            if self.onnx_ir_version > 9:
                expected_size = axes.size

            else:
                expected_size = len(ax)

        if starts and starts.size != expected_size:
            raise ValueError("Input 'starts' does not have correct amount of elements")

        if ends and ends.size != expected_size:
            raise ValueError("Input 'ends' does not have correct amount of elements")

        if steps and steps.size != expected_size:
            raise ValueError("Input 'steps' does not have correct amount of elements")

        # Default values are in place. Override with given values
        if axes:

            for i, d in enumerate(axes.data):
                if d < 0:
                    d = data.rank + d

                sta[d] = starts.data[i]
                en[d] = ends.data[i]

                if steps:
                    stp[d] = steps.data[i]

        elif self.onnx_ir_version > 9:

            for d in range(data.rank):
                sta[d] = starts.data[d]
                en[d] = ends.data[d]

                if steps:
                    stp[d] = steps.data[d]

        else:

            for i, d in enumerate(self.ax):
                if d < 0:
                    d = data.rank + d

                sta[d] = self.sta[i]
                en[d] = self.en[i]

                if steps:
                    stp[d] = 1

        res_shape: List[int] = []

        # Prune up corner cases: out of range indexing etc. and calculate output
        for d in range(data.rank):
            s = sta[d]
            e = en[d]
            st = stp[d]
            in_size = data.shape[d]

            if s < 0: s = in_size + s
            if e < 0: e = in_size + e
            if s > in_size: s = in_size
            if e > in_size: e = in_size

            sta[d] = s
            en[d] = e

            # From onnx2c
            # calculate the output dimension
            # ok, there probably exist a closed form for this algorithm.
            # but I'm tired :)
            if s > e:
                # Swap
                t = s
                s = e
                e = t

                s -= 1
                e -= 1

                if s < 0: s = 0
                if e > in_size: e = in_size

                st = -st

            num = 0
            for _ in range(s, e, st):
                num += 1

            res_shape.append(num)
            if num <= 0:
                #  https://github.com/onnx/onnx/issues/3724
                raise ValueError("Unimplemented: tensor sliced to have dimension of size 0")
        self.ax = ax
        self.sta = sta
        self.en = en
        self.stp = stp

        res = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=data.data.dtype))

        self.output = res

        self._register_output(self.output, "output")
        return [self.output]

    def print(self, destination: TextIO):
        destination.write("\t/* Slice */\n")

        out_idx = ""
        in_idx = ""

        # Loop over output dimensions & create the indexing arrays
        for d in range(self.output.rank):
            s = self.sta[d]
            e = self.en[d]
            st = self.stp[d]
            in_size = self.data.shape[d]

            # start and end have different semantics.
            # start index is inclusive, end exclusive.
            if s > e and s == in_size:
                s -= 1

            iv = f"i{d}"
            ov = f"o{d}"

            destination.write(
                f"\tfor (unsigned {iv} = {s}, {ov} = 0; "
                f"{ov} < {self.output.shape[d]}; "
                f"{iv} += {st}, ++{ov}) {{\n"
            )

            out_idx += f"[{ov}]"
            in_idx += f"[{iv}]"

        # Copy over data from input to output
        destination.write(f"\t\toutput{out_idx} = data{in_idx};\n")

        # close loops over output dimensions
        destination.write("\t}\n" * self.output.rank)


class Conv(SpatialFilter):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.auto_pad = "NOTSET"
        self.group = 1

        # Optional inputs
        self.b: tensor.Tensor = None

    def print_output_cell_init(self, destination: TextIO, y_idx: str = ""):
        outidx = ""
        for i in range(self.x.rank - 2):
            outidx += f"[o{i}]"

        destination.write(f"\t\t\tY[b][m]{outidx} = ")
        if self.b:
            destination.write(f"B[m];\n")
        else:
            destination.write("0;\n")

    def print_output_cell_calc(self, destination: TextIO, x_idx: str = "", w_idx: str = "", y_idx: str = ""):
        outidx = ""
        iididx = ""
        kidx = ""

        for i in range(self.x.rank - 2):
            outidx += f"[o{i}]"
            iididx += f"[ii{i}]"
            kidx += f"[k{i}]"

        destination.write(
            f"\t\t\t\tY[b][m]{outidx} += X[b][c]{iididx} * "
            f"W[m][c{'' if self.group == 1 else '-(gi*g)'}]{kidx};\n"
        )

    def print_output_cell_finalize(self, destination: TextIO, y_idx: str = ""):
        pass

    def print(self, destination: TextIO):
        self.print_header_info_comment(destination=destination)
        self.print_loop_with_padding_checks(destination=destination)

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        self.x = inputs[0]
        self.w = inputs[1]

        if len(inputs) == 3:
            self.b = inputs[2]
        else:
            self.b = None

        if not self.x.is_high_precision_numeric or not self.w.is_high_precision_numeric:
            raise ValueError("Incorrect input for node")
        if self.b and not self.b.is_high_precision_numeric:
            raise ValueError("Incorrect input for node")

        self.resolve_strides()
        self.resolve_dilations()
        self.resolve_pads()
        self.resolve_kernel_shape()

        res_shape = self.resolve_output_size()
        res = tensor.Tensor(data=np.ndarray(shape=res_shape, dtype=self.x.data.dtype))
        self.y = res

        self._register_input(self.x, "X")
        self._register_input(self.w, "W")
        if self.b:
            self._register_input(self.b, "B")
        self._register_output(self.y, "Y")

        return [self.y]


class BatchNormalization(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        # From onnx2c
        # TODO: is it ever possible that we can't compute sqrt(var) offline
        self.sqrt_var_offline = False

        self.epsilon = 1e-5
        self.momentum = 0.9

        self.input: tensor.Tensor = None    # 'X' in the spec
        self.scale: tensor.Tensor = None
        self.bias: tensor.Tensor = None    # 'B' in the spec
        self.mean: tensor.Tensor = None
        self.var: tensor.Tensor = None

        self.output: tensor.Tensor = None

        # TODO: implement optional outputs

    def parse_attributes(self):
        for a in self.onnx_node.attribute:
            if a.name == "epsilon":
                self.epsilon = utils.parse_attribute_float(a)
            elif a.name == "momentum":
                self.momentum = utils.parse_attribute_float(a)
            elif a.name == "spatial":
                # spatial was removed in opset v9
                spatial = utils.parse_attribute_int(a)
                if spatial != 1:
                    return ValueError("Non-default value for 'spatial' attribute not implemented")
            else:
                raise ValueError(f"Unimplemented parsing of attribute {a.name}")

    def print(self, destination: TextIO):
        batch_size, num_chan = self.input.shape[:2]
        dtype = self.input.data_type_str

        input_name = "input"
        mean_name = "mean"
        var_name = "var"
        out_name = "output"
        scale_name = "scale" if self.scale else None
        bias_name = "bias" if self.bias else None

        destination.write(
            f"\t/* BatchNormalization\n"
            f"\t * epsilon = {self.epsilon}\n"
            f"\t * momentum = {self.momentum}\n"
            f"\t */\n\n"
            f"\tfor (uint64_t b = 0; b < {batch_size}; ++b) {{\n"
            f"\tfor (uint64_t c = 0; c < {num_chan}; ++c) {{\n"
        )

        idxs = "[b][c]" + ''.join([f"[i{i}]" for i in range(2, self.input.rank)])

        for i, d in enumerate(self.input.shape[2:], 2):
            idx = f"i{i}"
            destination.write(f"\tfor (uint64_t {idx} = 0; {idx} < {d}; ++{idx}) {{\n")

        destination.write(f"\t\t{dtype} tmp_X = ( {input_name}{idxs} - {mean_name}[c] ) / ")

        if self.sqrt_var_offline:
            destination.write(f"( {var_name}[c] );\n")
        else:
            destination.write(f"( sqrt({var_name}[c] + {self.epsilon}) );\n")

        destination.write(f"\t\t{out_name}{idxs} = tmp_X")

        if self.scale:
            destination.write(f" * {scale_name}[c]")

        if self.bias:
            destination.write(f" + {bias_name}[c]")

        destination.write(";\n")
        destination.write("\t}\n" * self.input.rank)

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        if len(inputs) != 5:
            raise ValueError("incorrect number of inputs to BatchNormalization")

        if not all(map(lambda t: t.is_all_fp, inputs)):
            raise ValueError("incorrect input type for node")

        self.input, self.scale, self.bias, self.mean, self.var = inputs

        # It is possible for scale to be all ones and bias all zeros
        # scale and bias tensors aren't optional, so they're always available
        if utils.is_splatted(self.scale, 1):
            self.scale = None

        if utils.is_splatted(self.bias, 0):
            self.bias = None

        if self.var.isConst:
            self._calc_sqrt_var_offline()
            self.sqrt_var_offline = True

        self.output = tensor.Tensor(data=np.ndarray(shape=self.input.shape, dtype=self.input.data.dtype))

        self._register_input(self.input, "input")
        self._register_input(self.mean, "mean")
        self._register_input(self.var, "var")

        if self.scale:
            self._register_input(self.scale, "scale")

        if self.bias:
            self._register_input(self.bias, "bias")

        self._register_output(self.output, "output")

        return [self.output]

    def _calc_sqrt_var_offline(self):
        """
        Updates variance tensor in-place to contain the entire denominator
        of the BatchNormalization formula.
        """
        # From onnx2c
        # TODO: This breaks if var is used anywere else.

        self.var.data = np.sqrt(self.var.data + self.epsilon)


class Concat(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.axis: int = 1
        self.concat_result: tensor.Tensor = None
        self.node_inputs: List[tensor.Tensor] = []

    def parse_attributes(self):
        for a in self.onnx_node.attribute:
            if a.name == "axis":
                self.axis = utils.parse_attribute_int(a)

            else:
                raise ValueError(f"Unimplemented parsing of attribute {a.name}")

    def print(self, destination: TextIO):
        destination.write("\t/* Concat */\n")

        res_name = self.concat_result.cname
        dtype = self.concat_result.data_type_str
        res_shape: List[int] = self.concat_result.shape
        axis = self.axis

        # The axis_pitch is the number of elements to add to move to the next split axis in the concat_result
        axis_pitch = reduce(lambda x, y: x * y, res_shape[axis:])

        # TODO: this can technically overflow, which might be an issue with emitted results
        destination.write("\tuint64_t output_offset;\n")

        output_base = 0

        for node_input in self.node_inputs:

            # the input_axis_pitch is the number of elements to add to move to the next split axis in the inputs
            input_name = node_input.cname
            input_shape: List[int] = node_input.shape
            input_axis_pitch = reduce(lambda x, y: x * y, input_shape[axis:])
            input_size = node_input.size

            # copy the data across: for every 'input_axis_pitch' values copied, we move over by the 'axis_pitch'
            destination.write(
                f"\toutput_offset = {output_base};\n"
                f"\tfor (uint64_t i = 0, j = 0; i < {input_size}; ++i) {{\n"
                f"\t\t*(({dtype}*){res_name} + (output_offset + i)) = *(({dtype}*){input_name} + i);\n\n"
                f"\t\tif (++j == {input_axis_pitch}) {{\n"
                f"\t\t\toutput_offset += {axis_pitch - input_axis_pitch};\n"
                f"\t\t\tj = 0;\n"
                f"\t\t}}\n"
                f"\t}}\n"
            )

            output_base += input_axis_pitch

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        if not inputs:
            raise ValueError("Got empty inputs to Concat node")

        self.node_inputs = inputs

        if self.axis < 0:
            logging.debug(f"Got negative axis ({self.axis}) to concat node")
            self.axis += len(inputs[0].shape)
            logging.debug(f"New axis is {self.axis}")

        new_shape = inputs[0].shape
        output_axis_size = 0

        for node_input in inputs:
            for idx, d in enumerate(new_shape):
                if d != node_input.shape[idx] and idx != self.axis:
                    raise ValueError(
                        "Concat's input tensors must have the same shape, "
                        "except for the dimension size of the axis to concatenate on"
                    )

            output_axis_size += node_input.shape[self.axis]

        new_shape[self.axis] = output_axis_size
        self.concat_result = tensor.Tensor(data=np.ndarray(shape=new_shape, dtype=inputs[0].data.dtype))

        for idx, i in enumerate(self.node_inputs):
            self._register_input(i, f"input_{idx}")
        self._register_output(self.concat_result, "concat_result")

        return [self.concat_result]


class Constant(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.output: tensor.Tensor = None

    def parse_attributes(self):
        for a in self.onnx_node.attribute:
            logging.debug(f"Parsing attribute {a.name}")

            if a.name == "value":
                self.output = utils.parse_attribute_tensor(a)

            else:
                raise ValueError(f"Unimplemented parsing of attribute {a.name}")

    def print(self, destination: TextIO):
        destination.write(
            "\t/* Constant */\n"
            "\t/* The output is generated as a global tensor */\n"
            f"\t(void){self.output.cname};\n"
        )

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        if not self.output:
            raise RuntimeError("Constant output tensor was not resolved properly")

        self._register_output(self.output, "output")

        return [self.output]


class Reshape(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.data: tensor.Tensor = None
        self.shape: tensor.Tensor = None
        self.reshaped: tensor.Tensor = None

    def print(self, destination: TextIO):
        dtype = self.data.data_type_str

        # From onnx2c
        # TODO: is there ANY case where a reshape needs to re-order the internal data layout ?
        # TODO: and if not - check that at least gcc can get rid of this copy! (So onnx2c doesn't need to)
        # TODO: or - can we mark output an onnx2c-alias of input?

        destination.write(
            "\t/* Reshape */\n"
            f"\t{dtype} *data_ = ({dtype}*){self.data.cname};\n"
            f"\t{dtype} *reshaped_ = ({dtype}*){self.reshaped.cname};\n"
            f"\tfor(uint64_t i = 0; i < {self.data.size}; ++i)\n"
            "\t\treshaped_[i] = data_[i];\n\n"
        )

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        if len(inputs) != 2: raise RuntimeError("Unexpected number of inputs passed to Reshape node")

        data, shape = inputs

        if not shape.is_any_int:
            raise RuntimeError("Incorrect shape input type for node")

        if not shape.initialize:
            raise RuntimeError("Reshaping to a runtime defined shape is not supported")

        out_shape = []
        new_shape = shape.data
        negative_shape_found = False
        negative_shape_idx = -1

        output_size = 1
        for i, s in enumerate(new_shape):
            if s < 0:
                if negative_shape_found:
                    raise ValueError("Bad input: two negatives in reshape's target shape")

                negative_shape_found = True
                negative_shape_idx = i

            elif s == 0:
                if i >= len(data.shape):
                    raise ValueError("Bad input: Reshape requesting duplication of input dimension that doesn't exist")

                s = data.shape[i]

            out_shape.append(s)

            if s > 0:
                output_size *= s

        if negative_shape_found:
            missing_dim = data.size // output_size

            if output_size * missing_dim != data.size:
                raise ValueError("Could not deduce implicit dimension size for Resize node")

            out_shape[negative_shape_idx] = missing_dim

        reshaped = tensor.Tensor(data=np.ndarray(shape=out_shape, dtype=data.data.dtype))

        self.data = data
        self.shape = shape
        self.reshaped = reshaped

        self._register_input(self.data, "data")
        self._register_input(self.shape, "shape")
        self._register_output(self.reshaped, "reshaped")

        return [reshaped]


class Relu(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.X: tensor.Tensor = None
        self.Y: tensor.Tensor = None

    def print(self, destination: TextIO):
        X = self.X
        Y = self.Y
        dtype = X.data_type_str

        destination.write(
            "\t/* Relu */\n"
            f"\t{dtype} *X_ = ({dtype}*){X.cname};\n"
            f"\t{dtype} *Y_ = ({dtype}*){Y.cname};\n"
            f"\tfor(uint64_t i = 0; i < {X.size}; ++i)\n"
            "\t\tY_[i] = X_[i] > 0 ? X_[i] : 0;\n\n"
        )

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        X = inputs[0]

        if not (X.is_all_fp or X.is_signed_int):
            raise ValueError("Incorrect input for Relu")

        if X.shape[1] != 0 and not (X.shape[0] != 1 or X.shape[1] != 1):
            raise ValueError("Unsupported: multidim relu")

        Y = tensor.Tensor(data=np.ndarray(shape=X.shape, dtype=X.data.dtype))

        self.X = X
        self.Y = Y
        self._register_input(self.X, "X")
        self._register_output(self.Y, "Y")

        return [Y]


class Gemm(Node):
    def __init__(
        self,
        onnx_node: onnx.NodeProto = None,
        is_resolved: bool = False,
        op_name: str = None,
        onnx_name: str = None
    ) -> None:
        super().__init__(onnx_node, is_resolved, op_name, onnx_name)

        self.A: tensor.Tensor = None
        self.B: tensor.Tensor = None
        self.C: tensor.Tensor = None
        self.Y: tensor.Tensor = None

        self.alpha = 1.
        self.beta = 1.
        self.transA = 0
        self.transB = 0

    def parse_attributes(self):
        for a in self.onnx_node.attribute:
            logging.debug(f"Parsing attribute {a.name}")

            if a.name == "alpha":
                self.alpha = utils.parse_attribute_float(a)
            elif a.name == "beta":
                self.beta = utils.parse_attribute_float(a)
            elif a.name == "transA":
                self.transA = utils.parse_attribute_int(a)
            elif a.name == "transB":
                self.transB = utils.parse_attribute_int(a)
            else:
                raise ValueError(f"unknown attribute: {a.name}")

    def print(self, destination: TextIO):
        A = self.A
        B = self.B
        C = self.C
        Y = self.Y
        transA = self.transA
        transB = self.transB
        alpha = self.alpha
        beta = self.beta

        A1 = A.shape[1]
        C0 = 0
        C1 = 0
        if C:
            C0 = C.shape[0]
            if C.rank > 1:
                C1 = C.shape[1]

        M = A.shape[1] if transA else A.shape[0]    # row
        K = A.shape[0] if transA else A.shape[1]    # inner
        N = B.shape[0] if transB else B.shape[1]    # column
        dtype = A.data_type_str

        destination.write(
            f"\t/* Gemm */\n"
            f"\t/* alpha  = {alpha}\n"
            f"\t * beta   = {beta}\n"
            f"\t * transA = {transA}\n"
            f"\t * transB = {transB}\n"
            f"\t */\n"
            f"\tconst int M = {M};\n"
            f"\tconst int K = {K};\n"
            f"\tconst int N = {N};\n"
            f"\tfloat alpha = {alpha};\n"
            f"\tfloat beta = {beta};\n"
            f"\t{dtype} (*A_)[{A1}] = ({dtype}(*)[{A1}]){A.cname};\n"
            f"\t{dtype} (*Y_)[{N}] = ({dtype}(*)[{N}]){Y.cname};\n"
        )

        A_el = "A_[i][r]" if transA else "A_[r][i]"
        B_idx = "[c][i]" if transB else "[i][c]"

        # Cast optional C matrix to generated variable
        # "C[M][N]"
        if C:
            C_idx = ""
            if C.rank == 0:
                raise ValueError("Unimplemented: scalar C in Gemm")
            elif C.rank == 1:
                dim = C.shape[0]
                if dim == M:
                    C0 = M
                    C1 = 1
                elif dim == N:
                    C0 = 1
                    C1 = N
                elif dim == 1:
                    C0 = 1
                    C1 = 1
                else:
                    raise ValueError("C dimension mismatch in Gemm")
            elif C.rank == 2:
                C0, C1 = C.shape
            else:
                raise ValueError("C has too many dimensions in Gemm")

            C_idx += "[0]" if C0 <= 1 else "[r]"
            C_idx += "[0]" if C1 <= 1 else "[c]"

            destination.write(f"\t{dtype} (*C_)[{C1}] = ({dtype}(*)[{C1}]){C.cname};\n")

        # Now generate the calculation source code

        # Loop output rows, columns
        destination.write(
            "\tfor (uint32_t r = 0; r < M; ++r)\n"
            "\t\tfor (uint32_t c = 0; c < N; ++c) {\n"

        # Calculate the matrix multiplcation inner dot product
            f"\t\t\t{dtype} ABrc = 0;\n"
            "\t\t\tfor (uint32_t i = 0; i < K; ++i) {\n"
            f"\t\t\t\t{B.data_type_str} B_ = {B.cname}{B_idx};\n"
            f"\t\t\t\tABrc += {A_el} * B_;\n"
            "\t\t\t}\n"

        # Add scale & bias, store result in output
            f"\t\t\t{dtype} tmp = ABrc * alpha;\n"
        )

        if C:
            destination.write(f"\t\t\ttmp += C_{C_idx} * beta;\n")

        destination.write("\t\t\tY[r][c] = tmp;\n\t}\n")

    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        if len(inputs) < 2:
            raise ValueError("Not enough inputs")

        self.A, self.B = inputs[:2]

        if len(inputs) == 3:
            self.C = inputs[2]

        M = self.A.shape[1] if self.transA else self.A.shape[0]
        N = self.B.shape[0] if self.transB else self.B.shape[1]

        self.Y = tensor.Tensor(data=np.ndarray(shape=(M, N), dtype=self.A.data.dtype))

        self._register_input(self.A, "A")
        self._register_input(self.B, "B")
        if self.C:
            self._register_input(self.C, "C")
        self._register_output(self.Y, "Y")

        return [self.Y]


def from_onnx_node(onnx_node: onnx.NodeProto) -> Node:
    mapping = {
        "Add": Elementwise_2,
        "And": Elementwise_2,
        "AveragePool": AveragePool,
        "BatchNormalization": BatchNormalization,
        "BitShift": Elementwise_2,
        "Concat": Concat,
        "Constant": Constant,
        "Conv": Conv,
        "Div": Elementwise_2,
        "Equal": Elementwise_2,
        "Gemm": Gemm,
        "GlobalAveragePool": GlobalAveragePool,
        "Greater": Elementwise_2,
        "GreaterOrEqual": Elementwise_2,
        "Less": Elementwise_2,
        "LessOrEqual": Elementwise_2,
        "MatMul": MatMul,
        "MaxPool": MaxPool,
        "Mod": Elementwise_2,
        "Mul": Elementwise_2,
        "Or": Elementwise_2,
        "Pow": Elementwise_2,
        "PRelu": Elementwise_2,
        "Relu": Relu,
        "Reshape": Reshape,
        "Slice": Slice,
        "Sub": Elementwise_2,
        "Xor": Elementwise_2,
    }

    n = mapping[onnx_node.op_type](onnx_node)
    n.parse_attributes()

    return n
