import logging
from mimetypes import init
import onnx


class Graph:

    def __init__(self, onnx_model: onnx.ModelProto, ext_inputs=[]) -> None:
        self.onnx_model = onnx_model
        self.ext_inputs = ext_inputs
        self.tensors = []

    def process_graph(self):
        onnx_graph: onnx.GraphProto = self.onnx_model.graph

        # This might not be needed
        if self.ext_inputs:
            logging.debug("Adding external tensors")
        for ext_input in self.ext_inputs:
            logging.debug(f" - {ext_input.name}")
            self.tensors.append(ext_input)

        logging.debug("Adding initialized constant tensors")
        for initializer in onnx_graph.initializer:
            initializer: onnx.TensorProto

            self.add_initialized_tensor(initializer)

        logging.debug("Processing model inputs")
        for i in onnx_graph.input:
            i: onnx.ValueInfoProto

            t = self.get_io_tensor(i)
            self.tensors.append(t)

        logging.debug("Processing nodes")
        self.resolve_graph_nodes(onnx_graph)

        logging.debug("Processing model outputs")
        for o in onnx_graph.output:
            o: onnx.ValueInfoProto

            t = self.get_io_tensor(o)
            self.tensors.append(t)

    def add_initialized_tensor(self, t: onnx.TensorProto):
        ...

    def get_io_tensor(self, i: onnx.ValueInfoProto):
        return object()

    def resolve_graph_nodes(onnx_graph: onnx.GraphProto):
        ...
