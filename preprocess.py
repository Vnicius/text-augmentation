# -*- coding: utf-8 -*-
import os
import re
import pickle
import json
import collections
import numpy as np
from vlibras_translate.translation import Translation
from fastai.text import TextLMDataBunch, transform
from src.WordsDictionary import WordsDictionary
from src.args.PreprocessArgs import PreprocessArgs
from src.utils.utils import crate_dir

POINTS = re.compile(r'([\.\,\?\!])(\D|$)')
SYMBOLS = re.compile(r'([\(\)\[\]\{\}/;:])')
QUOTES = re.compile(r'\'([^\']+)\'|$')
DB_QUOTES = re.compile(r'\"([^\"]+)\"|$')
TREE_DOTS = re.compile(r'\.(\s+\.)+')
TAGS = re.compile(r'<[^>]+>')


def preprocess(text):
    processed_text = text.lower()
    words = []

    processed_text = TAGS.sub(r'', processed_text)
    processed_text = POINTS.sub(r' \1 \2', processed_text)
    processed_text = SYMBOLS.sub(r'', processed_text)
    processed_text = QUOTES.sub(r"\1", processed_text)
    processed_text = DB_QUOTES.sub(r'\1', processed_text)
    processed_text = TREE_DOTS.sub('.', processed_text)

    for word in processed_text.split(' '):
        striped = word.strip()
        if striped:
            words.append(striped)
    processed_text = " ".join(words)

    return processed_text


def generate_src_tgt(text):
    text_words = text.split(' ')

    for i in range(1, len(text_words)):
        yield((' '.join(text_words[:i]), text_words[i]))


def main(input_file, output_dir, max_size=0):
    crate_dir(output_dir)
    count = 0
    blank_limit = 1000
    blank_count = 0
    translator = Translation()

    with open(input_file, 'r', encoding='utf-8') as input_text, open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as output_train:
        count = 0
        for line in input_text:
            text, _ = translator.preprocess_train_files(line)
            text_size = len(text.split(' '))

            count += 1
            print(count)

            if text_size < 2:
                continue

            if max_size:
                if text_size > max_size:
                    text = ' '.join(text.split(' ')[:max_size])

            blank_count = 0
            output_train.write(text + '\n')
        # pickle.dump(words_dict, open(os.path.join(output_dir, 'data.dict'), 'wb'))

    print('\033[1;34m', 'Creating the Data Buntch', '\033[0;0m')
    phrases = []

    with open(os.path.join(output_dir, 'train.txt'), 'r', encoding='utf-8') as txt:
        phrases = [line.replace('\n', '').split(' ') for line in txt]

    freq = collections.Counter([w for s in phrases for w in s])

    max_vocab = 30000
    min_freq = 5

    # getting rid of the rare words
    itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')  # itos is the list of all the strings in the vocab

    stoi = collections.defaultdict(
        lambda: 0, {v: k for k, v in enumerate(itos)})
    max_data = int(len(phrases) * 0.9)
    # creating a index representation for our train and validation dataset

    trn_lm = np.array([[stoi[o] for o in p] for p in phrases])
    data_lm = TextLMDataBunch.from_ids(output_dir, transform.Vocab(
        itos), train_ids=trn_lm[:max_data], valid_ids=trn_lm[max_data:])

    #np.save(output_dir, trn_lm)
    #pickle.dump(itos, open(os.path.join(output_dir, 'itos.pkl'), 'wb'))
    #pickle.dump(dict(stoi), open(os.path.join(output_dir, 'stoi.pkl'), 'wb'))
    data_lm.save('data_save.pkl')


def preprocess_for_augmentation(input_file, output_dir, max_size=0):
    file_name = os.path.split(input_file)[-1]

    with open(input_file, 'r', encoding='utf-8') as input_text, open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as output_file, open(os.path.join(output_dir, 'prep_' + file_name), 'w', encoding='utf-8') as output_prep_file:
        translator = Translation()
        count = 0

        for line in input_text:
            txt = translator.preprocess_pt(line)

            if max_size:
                if len(txt.split(' ')) > max_size:
                    txt = txt[:max_size]

            output_file.write(txt + '\n')

            txt, _ = translator.preprocess_train_files(line)

            if max_size:
                if len(txt.split(' ')) > max_size:
                    txt = txt[:max_size]

            output_prep_file.write(txt + '\n')

            count += 1
            print(count)


if __name__ == '__main__':
    args = PreprocessArgs().args

    if args.for_augment:
        preprocess_for_augmentation(args.input_file, args.output, args.length)
    else:
        main(args.input_file, args.output, args.length)
    #dic = pickle.load(open('prep/dict.pkl', 'rb'))
    #zprint(word_to_vec('teste', dic))
