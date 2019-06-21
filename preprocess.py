# -*- coding: utf-8 -*-
import os
import re
import pickle
import json
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


def main(input_file, output_dir, max_size=0):
    crate_dir(output_dir)
    words_dict = WordsDictionary()
    infos = {}
    max_len = 0

    with open(input_file, 'r', encoding='utf-8') as input_text, open(os.path.join(output_dir, 'data.src'), 'w', encoding='utf-8') as output_src, open(os.path.join(output_dir, 'data.tgt'), 'w', encoding='utf-8') as output_tgt:
        try:
            while True:
                text = input_text.readline().replace("\n", '')

                if not len(text):
                    break
            
                text = preprocess(text)
                text_size = len(text.split(' '))

                if max_size:
                    if text_size > max_size:
                        text = ' '.join(text.split(' ')[:max_size])
                else:
                    if text_size > max_len:
                        max_len = text_size

                words_dict.add_text(text)               
                src, tgt = generate_src_tgt(text)

                for s, t in zip(src, tgt):
                    output_src.write(s + '\n')
                    output_tgt.write(t + '\n')
        except EOFError:
            pass

    words_dict.build_dict()
    pickle.dump(words_dict, open(os.path.join(output_dir, 'dact.dict'), 'wb'))

    with open(os.path.join(output_dir, 'data.meta.json'), 'w', encoding='utf-8') as infos_file:
        infos['max_seq_size'] = max_size if max_size else max_len

        json.dump(infos, infos_file)

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
