# -*- coding: utf-8 -*-
import os
import re
import pickle
from keras.utils import to_categorical
import numpy as np
from src.WordsDictionary import WordsDictionary
from src.args.PreprocessArgs import PreprocessArgs

POINTS = re.compile(r'([\.\,\?\!])(\D|$)')
SYMBOLS = re.compile(r'([\(\)\[\]\{\}])')
QUOTES = re.compile(r'\'([^\']+)\'|$')
DB_QUOTES = re.compile(r'\"([^\"]+)\"|$')
TREE_DOTS = re.compile(r'\.(\s+\.)+')


def crate_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def preprocess(text):
    processed_text = text.lower()
    words = []

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
    src = []
    tgt = []

    for i in range(1, len(text_words)):
        src.append(' '.join(text_words[:i]))
        tgt.append(text_words[i])

    return src, tgt


def main(input_file, output_dir):
    crate_dir(output_dir)
    words_dict = WordsDictionary()

    with open(input_file, 'r', encoding='utf-8') as input_text, open(os.path.join(output_dir, 'data.src'), 'w', encoding='utf-8') as output_src, open(os.path.join(output_dir, 'data.tgt'), 'w', encoding='utf-8') as output_tgt:
        try:
            while True:
                text = input_text.readline().replace("\n", '')

                if not len(text):
                    break

                text = preprocess(text)
                words_dict.add_text(text)
                src, tgt = generate_src_tgt(text)

                for s, t in zip(src, tgt):
                    output_src.write(s + '\n')
                    output_tgt.write(t + '\n')
        except EOFError:
            pass

    words_dict.build_dict()
    pickle.dump(words_dict, open(os.path.join(output_dir, 'dict.pkl'), 'wb'))


def text_2_vec(text, word_dict, max_size=0):
    text_vector = word_dict.text_to_int(text)
    vector = []

    if max_size:
        vector = np.zeros(max_size)
        vector[:len(text_vector)] = text_vector
    else:
        vector = text_vector

    return vector


def word_2_vec(word, word_dict):
    word_vec = [word_dict.word_to_int[word]]
    word_vec = to_categorical(
        word_vec, num_classes=len(word_dict.word_to_int))
    return word_vec


if __name__ == '__main__':
    #args = PreprocessArgs().args
    #main(args.input_file, args.output)
    #dic = pickle.load(open('prep/dict.pkl', 'rb'))
    #zprint(word_2_vec('teste', dic))
