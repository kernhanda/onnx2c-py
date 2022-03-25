#!/usr/bin/env python3

import argparse
import logging
import onnx

import graph

LOG_LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Generate C code from an ONNX graph file")

    arg_parser.add_argument("input", help="Path to the ONNX file", default=None)
    arg_parser.add_argument(
        "-v", "--verbosity", type=int, choices=range(len(LOG_LEVELS)), help="Sets the logging verbosity", default=2
    )
    arg_parser.add_argument("-l", "--log", help="Specifies the file to write log to", default="onnx2c.log")
    arg_parser.add_argument("output", help="Path to output file", default="-")

    args = vars(arg_parser.parse_args())
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=LOG_LEVELS[args["verbosity"]], filename=args["log"], filemode='w')

    g = graph.Graph(onnx.load(args["input"]))

    with open(args["output"], mode="wt", buffering=1) as f:
        g.print_source(f)


if __name__ == "__main__":
    main()
