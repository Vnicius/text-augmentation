# -*- coding: utf-8 -*-
import argparse


class PreprocessArgs():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('input_file', help='Input file path')
        parser.add_argument(
            '-o', '--output', help='Output directory', default='prep')
        parser.add_argument('-l', '--length', help='Max text length', default=0, type=int)
        parser.add_argument('--data-model', help='Create data model', action='store_true')
        self.args = parser.parse_args()
