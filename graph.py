import logging
import onnx
from typing import List

import node
import tensor


class Graph:

    def __init__(self, onnx_model: onnx.ModelProto, ext_inputs=[]) -> None:
        self.onnx_model = onnx_model
        self.ext_inputs = ext_inputs
        self.tensors: List[tensor.Tensor] = []
        self.nodes: List[onnx.NodeProto] = []

    def process_graph(self):
        onnx_graph: onnx.GraphProto = self.onnx_model.graph

        # This might not be needed
        if self.ext_inputs:
            logging.info("Adding external tensors")
        for ext_input in self.ext_inputs:
            logging.info(f" - {ext_input.name}")
            self.tensors.append(ext_input)

        logging.info("Adding initialized constant tensors")
        for initializer in onnx_graph.initializer:
            initializer: onnx.TensorProto

            self.add_initialized_tensor(initializer)

        logging.info("Processing model inputs")
        for i in onnx_graph.input:
            i: onnx.ValueInfoProto

            self.add_io_tensor(i)

        logging.info("Processing nodes")
        self.resolve_graph_nodes(onnx_graph)

        logging.info("Processing model outputs")
        for o in onnx_graph.output:
            o: onnx.ValueInfoProto

            self.add_io_tensor(o)

    def add_initialized_tensor(self, t: onnx.TensorProto):
        t = tensor.parse_onnx_tensor(t)
        self.add_tensor(t)

    def add_io_tensor(self, vi: onnx.ValueInfoProto):
        t = tensor.parse_onnx_value_info(vi)
        self.tensors.append(t)

    def resolve_graph_nodes(self, onnx_graph: onnx.GraphProto):
        nodes: List[onnx.NodeProto] = onnx_graph.node[:]
        num_unresolved = len(nodes)
        num_resolved = 0

        while num_resolved < num_unresolved:

            for idx, node in enumerate(nodes[:]):

                if self.try_resolve_node(node):
                    nodes.pop(idx)

            num_resolved = num_unresolved - len(nodes)

            if num_resolved == 0:
                raise RuntimeError("Failed to resolve nodes")

            if num_resolved == num_unresolved:
                break

            num_resolved = 0
            num_unresolved = len(nodes)

    def try_resolve_node(self, onnx_node: onnx.NodeProto) -> bool:
        logging.info(f"Resolving node {onnx_node.name}")

        for n in self.nodes:
            if onnx_node.name == n.name:
                return True

        inputs = self.get_node_input_tensors(onnx_node)
        if inputs is None:
            return False

        c_node = node.create_node(onnx_node)
        ...

    def get_node_input_tensors(self, node: onnx.NodeProto):

        resolved: List[tensor.Tensor] = []

        for i in node.input:

            input_resolved = False

            # unused input, doesn't need to be resolved
            if not i:
                input_resolved = True
            else:
                for t in self.tensors:
                    if t.name == i:
                        input_resolved = True
                        resolved.append(t)
                        break

            if not input_resolved:
                logging.info(f"Input tensor {i} not resolved")
                return None

        else:

            return resolved

    def add_tensor(self, t: tensor.Tensor):
        existing_tensor = None
        for o in self.tensors:
            if o.name == t.name:
                existing_tensor = o
                break

        if not existing_tensor:
            self.tensors.append(t)
            logging.info(f"Adding new tensor: {t.name} - {t.data_type_str} - {t.str_dimensions}")

        else:
            logging.info(f"Updating existing tensor: {t.name}")

            logging.debug(
                f"\twas: gen {existing_tensor.generate} "
                f" init {existing_tensor.initialize} "
                f" IO {existing_tensor.isIO} "
                f" const {existing_tensor.isConst} "
                f" recurs {existing_tensor.isRecursive}"
            )

            logging.debug(
                f"\tnew: gen {t.generate} "
                f" init {t.initialize} "
                f" IO {t.isIO} "
                f" const {t.isConst} "
                f" recurs {t.isRecursive}"
            )

            if t.isRecursive:
                # Since this tensor was already added, it was added because it as a graph output
                # This is because recursion means recursion to same node, not a general loop in the network

                if not existing_tensor.isIO:
                    raise RuntimeError("Update logic failure")

                existing_tensor.generate = t.generate
                existing_tensor.initialize = t.initialize
                existing_tensor.isRecursive = True

            # Comment from onnx2c: Recursive nodes might need to initialize internal tensors
            # TODO: Should this be moved inside the isRecursive if check?
            if t.data:
                existing_tensor.data = t.data

            existing_tensor.initialize = existing_tensor.initialize or t.initialize

            if existing_tensor.initialize:
                existing_tensor.generate = True

            # Comment from onnx2c: huh? what is this use case?
            if not existing_tensor.isIO:
                existing_tensor.isConst = t.isConst

            # Comment from onnx2c:
            # The case where a tensor is marked to be IO.
            # The graph has a lists of input tensors that are input to the first
            # node, not necessarily an input from the user.
            # If the user doesn't provide them, they must be initialized in the graph.
            # E.g. the weights for a Convolution at the start is such an example
            if t.isIO and not existing_tensor.initialize:
                existing_tensor.isIO = True

            logging.debug(
                f"\tnow: gen {existing_tensor.generate} "
                f" init {existing_tensor.initialize} "
                f" IO {existing_tensor.isIO} "
                f" const {existing_tensor.isConst} "
                f" recurs {existing_tensor.isRecursive}"
            )
