# -*- coding: utf-8 -*-
import argparse


class AugmentArgs():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'original_data', help='Original data file', type=str)
        parser.add_argument(
            'original_data_preprocessed', help='Original data file with preprocess', type=str)
        parser.add_argument(
            'model_path', help='Path of the model file', type=str)
        parser.add_argument(
            'output_file', help='Output file with the augmented data', type=str)
        parser.add_argument(
            '--max-freq', help='Max frequence needed to a word be augmented', type=int, default=10)
        parser.add_argument(
            '--top-k', help='Top K words considered in the augmentations', type=int, default=5)
        parser.add_argument(
            '--n-augment', help='Number of augmentations of a word', type=int, default=5)
        self.args = parser.parse_args()
