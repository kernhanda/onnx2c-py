import datetime
import logging
import sys
import onnx
from typing import List, TextIO

import node
import tensor


class Graph:
    def __init__(self, onnx_model: onnx.ModelProto, ext_inputs=[]) -> None:
        self.onnx_model = onnx_model
        self.ext_inputs = ext_inputs
        self.tensors: List[tensor.Tensor] = []
        self.nodes: List[node.Node] = []

    def print_header(self, destination: TextIO):
        self.print_file_frontmatter(destination)

    def print_source(self, destination: TextIO):
        self.print_file_frontmatter(destination)
        destination.write("\n")
        self.print_includes(destination)
        destination.write("\n")
        self.print_global_tensors(destination)
        destination.write("\n")
        self.print_functions(destination)
        destination.write("\n")
        self.print_interface_function(destination)

    def print_file_frontmatter(self, destination: TextIO):
        print("// This file is generated by onnx2c-py", file=destination)
        print("// " + " ".join(sys.argv), file=destination)
        print(f"// {datetime.datetime.now()}\n", file=destination)

        print("// ONNX model:", file=destination)
        print(
            f"// produced by {self.onnx_model.producer_name}, "
            f"version {self.onnx_model.producer_version}",
            file=destination
        )
        print(f"// ONNX IR version: {self.onnx_ir_version}", file=destination)

        if self.onnx_model.doc_string:
            print(f"// Model documentation:", file=destination)
            print("/*", file=destination)
            print(self.onnx_model.doc_string)
            print("*/", file=destination)

    def print_global_tensors(self, destination: TextIO):
        for t in self.tensors:
            if not t.data.shape:
                raise RuntimeError("Trying to print a tensor with no dimensions")

            if not t.generate: continue
            if t.isIO: continue

            if len(t.data.shape) == 1 and t.data.shape[0] == 0:
                logging.warn(f"Tensor {t.name} has size of 0. Skipping it")
                continue

            destination.write(f"/* {t.name} */\n")
            destination.write("static ")
            t.print_tensor(destination)
            if t.initialize:
                destination.write(" = ")
                t.print_tensor_initializer(destination=destination)
            destination.write(";\n")

    def print_functions(self, destination: TextIO):
        for n in self.nodes:
            destination.write(f"static inline void {n.c_name}( ")
            n.print_func_params_shapes(destination)
            destination.write(" )\n{\n")
            n.print(destination)
            destination.write("}\n\n")

    def print_includes(self, destination: TextIO):
        headers = ["float.h", "math.h", "stdbool.h", "stdint.h", "string.h"]
        for h in headers:
            destination.write(f"#include <{h}>\n")

        destination.write(
            "#define MAX(X,Y) ( X > Y ? X : Y)\n"
            "#define MIN(X,Y) ( X < Y ? X : Y)\n"
            "#define CLIP(X,L) ( MAX(MIN(X,L), -L) )\n\n"
        )

    def print_interface_function(self, destination: TextIO, interface_name: str = None):
        is_first = True
        interface_name = interface_name or self.onnx_model.graph.name

        destination.write(f"void {interface_name}(")

        for i in self.onnx_model.graph.input:
            i: onnx.ValueInfoProto

            t = self.find_tensor(i.name)

            if t:
                if not is_first:
                    destination.write(", ")
                else:
                    is_first = False

                t.print_tensor_as_const(destination)

        for o in self.onnx_model.graph.output:
            o: onnx.ValueInfoProto

            t = self.find_tensor(o.name)

            if t:
                if not is_first:
                    destination.write(", ")
                else:
                    is_first = False

                t.print_tensor(destination)

        destination.write(") {\n")

        for n in self.nodes:
            destination.write(f"\t{n.c_name}( ")
            n.print_func_params_callsite(destination)
            destination.write(f");")

        destination.write("}\n")

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
        # TODO: check if add_tensor can be called on t instead
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

        n: node.Node = node.create_node(onnx_node)

        outputs = n.resolve_node(inputs)

        for idx, t in enumerate(outputs):
            onnx_name = onnx_node.output[idx] if n.is_output_used(idx) else ""

            if not onnx_name:
                if not t.isRecursive:
                    logging.debug(f'skipping: output number {idx} is unused')
                    continue

                onnx_name = f"{n.c_name}_recursive_{idx}"

            t.name = onnx_name

            self.add_tensor(t)

        n.is_resolved = True
        self.nodes.append(n)

        logging.info(f"Adding {n.op_name} node: {n.onnx_name}")
        logging.info("\tinputs:")
        logging.info(f"\t{' | '.join([i.name for i in inputs])}")
        logging.info("\toutputs:")
        logging.info(f"\t{' | '.join([o.name for o in outputs])}")

        return True

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

    @property
    def has_unresolved_nodes(self) -> bool:
        return len(self.onnx_model.graph.node) > len(self.nodes)

    @property
    def onnx_ir_version(self):
        if len(self.onnx_model.opset_import) > 1:
            raise RuntimeError("Unsupported: Model has multiple opset versions")

        return self.onnx_model.opset_import[0].version

    def find_tensor(self, name: str):
        for t in self.tensors:
            if t.name == name:
                return t

        return None
