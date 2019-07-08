# -*- condig: utf-8 -*-
from os import path
import pickle
import json
from random import shuffle
import numpy as np
from fastai.text import *
from src.utils.utils import crate_dir
from src.args.TrainArgs import TrainArgs


def train(model, epochs):

    return model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)


def main(data_dir, output_dir, model_dir, epochs):

    crate_dir(output_dir)

    data = load_data(data_dir, 'data_save.pkl')

    model = language_model_learner(data, text.models.AWD_LSTM, drop_mult=0.5)

    try:
        print("Loading checkpoint!")
        model.load_encoder(model_dir)
    except FileNotFoundError:
        print("No checkpoint!")
        pass

    model.fit(epochs)

    model.save_encoder(model_dir)


if __name__ == '__main__':
    args = TrainArgs().args

    main(args.data_dir, args.output, model_dir=args.model_dir, epochs=args.epochs)
