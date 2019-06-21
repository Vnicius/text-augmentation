# -*- coding: utf-8 -*-
import argparse


class PreprocessArgs():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('input_file', help='Input file path')
        parser.add_argument(
            '-o', '--output', help='Output directory', default='prep')
        self.args = parser.parse_args()
