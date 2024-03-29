# -*- coding: utf-8 -*-
from os import path
import re
from collections import Counter
import numpy as np
from fastai.text import *
import torch
from vlibras_translate.translation import Translation
from src.args.AugmentArgs import AugmentArgs
from src.AugmentationItem import AugmentationItem
from src.RareWord import RareWord

#l = load_learner('prep', 'model.pkl')


def get_dir_and_file(file_path):
    path_split = path.split(file_path)
    file_name = path_split[-1]
    file_dir = path.join(*path_split[:-1])

    return file_dir, file_name


def filter_rare_words(rare_words):
    words = []

    for w in rare_words:
        if w not in ["'", '@', '#', '"']:
            words.append(w)

    return words


def bests(learner, text: str, topk: int, n_words: int = 1, no_unk: bool = True, temperature: float = 0.5, min_p: float = None, sep: str = ' ',
          decoder=decode_spec_tokens):

    ds = learner.data.single_dl.dataset
    learner.model.reset()
    xb, yb = learner.data.one_item(text)
    new_idx = []
    res = learner.pred_batch(batch=(xb, yb))[0][-1]

    if no_unk:
        res[learner.data.vocab.stoi[UNK]] = 0.
    if min_p is not None:
        if (res >= min_p).float().sum() == 0:
            warn(
                f"There is no item with probability >= {min_p}, try a lower value.")
        else:
            res[res < min_p] = 0.
    if temperature != 1.:
        res.pow_(1 / temperature)
    idxs = torch.multinomial(res, topk)

    return [t.item() for t in idxs]


def has_augmentation(augmentations, sentence_id, position, word_id):
    for a in augmentations:
        if a.sentence_id == sentence_id and a.position == position and a.word_id == word_id:
            return True

    return False

def is_valid_word(word):
    if re.match(r'\w+', word) and not re.match(r'\d+', word):
        return True
    
    return False

def main(original_data, original_data_prep, model_path, output_file, max_freq, topk, n_augment, max_length=0):
    augmentations = []
    model_dir, model_name = get_dir_and_file(model_path)
    translator = Translation()

    print('\033[1;34m', 'Loading model', '\033[0;0m')
    model = load_learner(model_dir, model_name)

    print('\033[1;34m', 'Loading dict', '\033[0;0m')
    vocab = model.data.vocab

    print('\033[1;34m', 'Getting original data', '\033[0;0m')
    original = []
    original_prep = []
    max_size = [] 

    with open(original_data, 'r', encoding='utf-8') as od:
        for line in od:
            original.append(line.replace('\n', '').split(' '))

    with open(original_data_prep, 'r', encoding='utf-8') as prep_od:
        for line in prep_od:
            original_prep.append(line.replace('\n', '').split(' '))

    max_size = max_length if max_length else max([len(s) for s in original])

    print('\033[1;34m', 'Counting frequences', '\033[0;0m')
    
    freq = Counter([word for line in original_prep for word in line])
    
    print('\033[1;34m', 'Getting rare words', '\033[0;0m')

    rare_words = [k for k, v in freq.most_common(30000) if v <= max_freq]
    rare_words = filter_rare_words(rare_words)

    rare_words_ids = [vocab.stoi[w] for w in rare_words]
    rare_words_ids = [RareWord(i, vocab.itos[i])
                      for i in rare_words_ids if i != 0]

    print('\033[1;34m', f'{len(rare_words_ids)} Rare words', '\033[0;0m')

    print('\033[1;34m', f'Making augmentation', '\033[0;0m')
    was_augmented = True
    max_augment_lines = len(original) * n_augment

    with open(output_file, 'w', encoding='utf-8') as out:
        while was_augmented and len(augmentations) < max_augment_lines:
            for i in range(1, max_size):
                was_augmented = False
                for index, sent in enumerate(original_prep):
                    if i < len(sent) and is_valid_word(original[index][i]):
                        top_words = bests(model, ' '.join(sent[:i]), topk)

                        for r in rare_words_ids:
                            if r.id in top_words and not has_augmentation(augmentations, index, i, r.id) and r.occurrences < n_augment:
                                augmented = original[index][:]
                                augmented[i] = vocab.itos[r.id]
                                augmentations.append(
                                    AugmentationItem(index, i, r.id))
                                out.write(' '.join(augmented) + '\n')
                                r.occurrences += 1
                                was_augmented = True


if __name__ == '__main__':
    args = AugmentArgs().args

    main(original_data=args.original_data, original_data_prep=args.original_data_preprocessed, model_path=args.model_path, output_file=args.output_file,
         max_freq=args.max_freq, topk=args.top_k, n_augment=args.n_augment, max_length=args.max_length)
