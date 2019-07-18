# -*- condig: utf-8 -*-
from os import path
import pickle
import json
import collections
import glob
from random import shuffle
import re
import numpy as np
from fastai.text import *
from fastai.callbacks.tracker import SaveModelCallback
from fastai.metrics import accuracy
from src.utils.utils import crate_dir
from src.args.TrainArgs import TrainArgs


def get_last_epoch(checkpoint_dir):
    checkpoints = glob.glob(path.join(checkpoint_dir, '*.pth'))
    values = []

    for check in checkpoints:
        match = re.search(r'check_(\d+)', check)

        if match:
            values.append(int(match.group(1)))

    values.sort()
    return values[-1] + 1 if values else 0


def train(model, epochs):

    return model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)


def main(data_dir, output_dir, checkpoint_dir, epochs):

    crate_dir(output_dir)

    data = load_data(data_dir, 'data_save.pkl')

    model = language_model_learner(data, text.models.AWD_LSTM, drop_mult=0.5)

    try:
        print("Loading checkpoint!")
        model.load_encoder(checkpoint_dir)
    except FileNotFoundError:
        print("No checkpoint!")
        pass

    for _ in range(epochs):
        model.unfreeze()
        model.fit(1)
        model.save_encoder(checkpoint_dir)


def main2(data_dir, output_dir, epochs):

    crate_dir(output_dir)

    em_sz, nh, nl = 400, 1150, 3
    wd = 1e-7
    bptt = 70
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    bs = 32
    lr = 1e-3

    last_epoch = get_last_epoch(path.join(args.data_dir, 'models'))

    print('\033[1;34m', 'Loading data', '\033[0;0m')
    data = load_data(data_dir, 'data_save.pkl')
    model = language_model_learner(data, text.models.AWD_LSTM, drop_mult=0.5, metrics=[accuracy])

    try:
        print('\033[1;34m', 'Loading checkpoint', '\033[0;0m')
        model.load("last")
        print('\033[0;32m', 'Loaded last checkpoint', '\033[0;0m')
    except FileNotFoundError:
        print('\033[1;31m', 'No checkpoint founded', '\033[0;0m')
        pass

    
    model.fit(epochs, lr=slice(lr / 2.6, lr), wd=1e-7, callbacks=[SaveModelCallback(model, every='epoch', monitor='accuracy', name=f'check_{last_epoch}')])

    print('\033[0;32m', 'Saving model', '\033[0;0m')
    model.save("last")
    model.export("model.pkl")


if __name__ == '__main__':
    args = TrainArgs().args

    main2(args.data_dir, args.output, epochs=args.epochs)
