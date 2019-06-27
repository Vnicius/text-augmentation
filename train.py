# -*- condig: utf-8 -*-
from os import path
import pickle
import json
from random import shuffle
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Activation, Embedding, Bidirectional
from keras.callbacks import ModelCheckpoint
from preprocess import text_to_vec, preprocess, word_to_vec, generate_src_tgt
from src.utils.utils import crate_dir
from src.args.TrainArgs import TrainArgs


def create_model(dict_len, max_seq_len, embedding_size=32, lstm_size=128):
    model = Sequential()
    model.add(Embedding(dict_len, embedding_size, input_length=max_seq_len))
    model.add(Bidirectional(LSTM(lstm_size)))
    model.add(Dense(dict_len))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=['accuracy'])
    return model

def load_model(model_dir):
    json_file = open(path.join(model_dir,'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path.join(model_dir,"model.h5"))
    loaded_model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=['accuracy'])

    return loaded_model

# def train(model, x_data, y_data, batch_size, epochs):
#     return model.fit(x=x_data, y=y_data, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, shuffle=True)


def train_generator(model, generator, steps_per_epoch, epochs, callbacks):
    
    return model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)


def generate_arrays_from_file(path_train_data, word_dict, meta):
    while True:
        with open(path_train_data) as data_train:
            for line in data_train:
                data = list(generate_src_tgt(line.replace('\n', '')))
                shuffle(data)

                for d in data:
                    x = text_to_vec(preprocess(d[0]), word_dict, meta['max_seq_size'])
                    y = word_to_vec(preprocess(d[1]), word_dict)

                    yield (np.array([x]), np.array(y))


def main(data_dir, output_dir, model_dir, embedding_size, lstm_size, steps_per_epoch, epochs):
    crate_dir(output_dir)

    with open(path.join(data_dir, 'data.dict'), 'rb') as data_dict, open(path.join(data_dir, 'data.meta.json')) as meta_file:
        meta = json.load(meta_file)
        word_dict = pickle.load(data_dict)
        model = None

        if model_dir:
            model = load_model(model_dir)
        else:
            model = create_model(meta['dict_size'], meta['max_seq_size'],
                             embedding_size=embedding_size, lstm_size=lstm_size)

        # checkpoint
        filepath="weights-improvement-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(path.join(output_dir, filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        train_generator(model, generate_arrays_from_file(path.join(data_dir, 'data.train'), word_dict, meta), steps_per_epoch, epochs, callbacks_list)

        json_model = model.to_json()

        with open(path.join(output_dir, 'model.json'), "w") as json_file:
            json_file.write(json_model)

        model.save_weights(path.join(output_dir, 'model.h5'))


if __name__ == '__main__':
    args = TrainArgs().args

    main(args.data_dir, args.output, model_dir=args.model_dir, embedding_size=args.embedding_size,
         lstm_size=args.lstm_size, steps_per_epoch=args.steps, epochs=args.epochs)
