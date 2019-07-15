# -*- coding: utf-8 -*-


class AugmentationItem():
    def __init__(self, sentence_id, position, word_id):
        self.sentence_id = sentence_id
        self.position = position
        self.word_id = word_id
    
    def __repr__(self):
        return f'({self.sentence_id}, {self.position}, {self.word_id})'
