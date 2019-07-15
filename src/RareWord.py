# -*- coding: utf-8 -*-


class RareWord():
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.occurrences = 0

    def __repr__(self):
        return f'({self.id}, {self.text}, {self.occurrences})'
