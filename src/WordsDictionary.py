# -*- conding: utf-8 -*-
from collections import OrderedDict


class WordsDictionary():
    def __init__(self):
        self.words_list = {}
        self.word_to_int = {'<PAD>': 0, '<UNK>': 1}
        self.int_to_word = {0: '<PAD>', 1: '<UNK>'}

    def add_text(self, text):
        for word in text.split(' '):
            try:
                self.words_list[word] += 1
            except KeyError:
                self.words_list[word] = 0

    def build_dict(self, limit=0):
        self.words_list = dict(OrderedDict(self.words_list.items()))

        for word in self.words_list:
            if limit and len(self.word_to_int) == limit:
                break

            self.word_to_int[word] = len(self.word_to_int) + 1
            self.int_to_word[len(self.int_to_word) + 1] = word

    def text_to_int(self, text):
        vector = []

        for word in text.split(' '):
            try:
                vector.append(self.word_to_int[word])
            except KeyError:
                vector.append(self.word_to_int['<UNK>'])

        return vector

    def int_vector_to_text(self, vector):
        text = []

        for number in vector:
            if number != self.word_to_int['<PAD>']:
                text.append(self.int_to_word[number])

        return ' '.join(text)
