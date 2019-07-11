# -*- condig: utf-8 -*-
from os import path
import pickle
import json
import collections
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

    for _ in range(epochs):
        model.unfreeze()
        model.fit(1)
        model.save_encoder(model_dir)

def main2(data_dir, output_dir, model_dir, epochs):

    crate_dir(output_dir)
    phrases = []

    with open(path.join(data_dir, 'train', 'train.txt'), 'r', encoding='utf-8') as txt:
        phrases = [s for line in txt for s in line.replace('\n', '').split()]
    
    freq = collections.Counter([w for s in phrases for w in s]) 

    max_vocab = 30000
    min_freq = 5

    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq] # getting rid of the rare words
    itos.insert(0, '_pad_') # 
    itos.insert(0, '_unk_') # itos is the list of all the strings in the vocab

    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

    # creating a index representation for our train and validation dataset
    trn_lm = np.array([[stoi[o] for o in p] for p in phrases])

    em_sz,nh,nl = 400,1150,3
    wd=1e-7
    bptt=70
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    bs = 32

    data = TextLMDataBunch.from_ids(data_dir, text.transform.Vocab(itos), train_ids=trn_lm[:int(len(trn_lm))], valid_ids=trn_lm[int(len(trn_lm)):])

    model = language_model_learner(data, text.models.AWD_LSTM, drop_mult=0.5)

    try:
        print("Loading checkpoint!")
        model.load_encoder(model_dir)
    except FileNotFoundError:
        print("No checkpoint!")
        pass

    for _ in range(epochs):
        model.unfreeze()
        model.fit(1)
        model.save_encoder(model_dir)


if __name__ == '__main__':
    args = TrainArgs().args

    main2(args.data_dir, args.output, model_dir=args.model_dir, epochs=args.epochs)
