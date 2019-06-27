import argparse


class TrainArgs():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('data_dir', help="Data directory", type=str)
        parser.add_argument(
            '-o', '--output', help="Output directory", type=str, default='model')
        parser.add_argument('--embedding_size',
                            help='Embedding Size', type=int, default=64)
        parser.add_argument('--lstm_size', help="LSTM size",
                            type=int, default=128)
        parser.add_argument(
            '-s', '--steps', help="Steps per epoch", type=int, default=10000)
        parser.add_argument(
            '-e', '--epochs', help="Number of epochs", type=int, default=10)
        parser.add_argument('-m', '--model_dir', help='Model directory', type=str, default='')
        self.args = parser.parse_args()
