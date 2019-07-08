# -*- coding: utf-8 -*-
import os
import re
import pickle
import json
import numpy as np
from fastai.text import TextLMDataBunch
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
    train_path = os.path.join(output_dir, 'train')

    crate_dir(output_dir)
    crate_dir(train_path)
    count = 0
    blank_limit = 5
    blank_count = 0

    with open(input_file, 'r', encoding='utf-8') as input_text, open(os.path.join(train_path, 'train.txt'), 'w', encoding='utf-8') as output_train:
        try:
            while True:
                text = input_text.readline().replace("\n", '')

                count += 1
                print(count)

                text = preprocess(text)
                text_size = len(text.split(' '))

                if text_size == 1:
                    blank_count += 1

                if blank_count == blank_limit:
                    break

                if text_size < 5:
                    continue

                if max_size:
                    if text_size > max_size:
                        text = ' '.join(text.split(' ')[:max_size])

                blank_count = 0
                output_train.write(text + "\n")

        except EOFError:
            pass

        data_lm = TextLMDataBunch.from_folder(
            output_dir, train='train', max_vocab=30000)

        data_lm.save('data_save.pkl')


def text_to_vec(text, word_dict, max_size=0):
    text_vector = word_dict.text_to_int(text)
    vector = []

    if max_size:
        vector = np.zeros(max_size)
        vector[:len(text_vector)] = text_vector
    else:
        vector = text_vector

    return vector


def vec_to_text(vector, word_dict):
    return word_dict.int_to_text(vector)


def word_to_vec(word, word_dict):
    word_vec = [word_dict.word_to_int[word]]
    word_vec = to_categorical(
        word_vec, num_classes=len(word_dict.word_to_int))
    return word_vec


def word_oh_to_int(word_oh):
    return np.argmax(word_oh)


if __name__ == '__main__':
    args = PreprocessArgs().args
    main(args.input_file, args.output, args.length)
    #dic = pickle.load(open('prep/dict.pkl', 'rb'))
    #zprint(word_to_vec('teste', dic))
