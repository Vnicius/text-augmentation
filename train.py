# -*- condig: utf-8 -*-
from os import path
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Embedding, Bidirectional 
from preprocess import text_to_vec, preprocess

def create_model(dict_len, max_seq_len, embedding_size=32, lstm_size=128):
    model = Sequential()
    model.add(Embedding(dict_len, embedding_size, input_length=max_seq_len))
    model.add(Bidirectional(LSTM(lstm_size)))
    model.add(Dense(dict_len))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

def train(model, x_data, y_data, batch_size, epochs):
    return model.fit(x=x_data, y=y_data, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, shuffle=True)

def main(data_dir, output_dir, embedding_size, lstm_size, batch_size, epochs):
    with open(path.join(data_dir, 'data.src'), 'r', encoding='utf-8') as src_data, open(path.join(data_dir, 'data.tgt'), 'r', encoding='utf-8') as tgt_data, open(path.join(data_dir, 'data.dict'), 'rb') as data_dict, open(path.join(data_dir, 'data.meta.json')) as meta_file:
        meta = json.load(meta_file)
        word_dict = pickle.load(data_dict)
        x_data = []

        try:
            while True:
                x_data.append(text_to_vec(preprocess(src_data.readline()), word_dict,meta['max_seq_size']))
        except EOFError:
            pass
        
        x_data = np.array(x_data)

        print(x_data.shape)



if __name__ == '__main__':
    main('prep', 'out', 32, 128, 2, 2)