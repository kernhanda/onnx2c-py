from abc import ABC, abstractmethod
from typing import IO, List, Tuple

import onnx

import nodes
import tensor
import utils

_anonymous_node_counter = 0

FuncParam = Tuple[tensor.Tensor, str]


class Node(ABC):

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

        if self.onnx_node:
            self.parse_atstributes(self.onnx_node)

    def parse_attributes(self, onnx_node: onnx.NodeProto):
        ...

    @abstractmethod
    def print(self, destination: IO):
        ...

    def print_func_params(self, destination: IO, decorate: bool):
        ...

    def print_func_params_shapes(self, destination: IO):
        ...

    def print_func_params_callsite(self, destination: IO):
        ...

    @abstractmethod
    def resolve_node(self, inputs: List[tensor.Tensor]) -> List[tensor.Tensor]:
        ...

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

    def parse_attributes(self, onnx_node: onnx.NodeProto):
        raise ValueError(f"Attribute parsing not implemented for node operation type {self.op_name}")

    @property
    def c_name(self):
        return "node_" + utils.cify_name(self.onnx_name)

    def multidirectional_broadcast_size(self, A: List[int], B: List[int]) -> List[int]:
        ...

    def _register_input(self, t: tensor.Tensor, name: str):
        ...

    def _register_output(self, t: tensor.Tensor, name: str):
        ...


def from_onnx_node(onnx_node: onnx.NodeProto) -> Node:
    return Node()
