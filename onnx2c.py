#!/usr/bin/env python3

import argparse
import onnx

import graph

def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Generate C code from an ONNX graph file")

    arg_parser.add_argument("input",
                            help="Path to the ONNX file",
                            default=None)
    arg_parser.add_argument("output", help="Path to output file")

    args = vars(arg_parser.parse_args())
    return args


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
