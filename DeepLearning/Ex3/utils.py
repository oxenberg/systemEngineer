import gensim.downloader
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional, BatchNormalization, TimeDistributed, RNN, LSTMCell
from tensorflow import keras
from tensorflow import convert_to_tensor, concat
from pretty_midi import PrettyMIDI
from tensorflow.keras.optimizers import Adam
from mido.midifiles.meta import KeySignatureError
import pickle

params = {
    "TEST_FILE": "lyrics_test_set.csv",
    "TRAIN_FILE": "lyrics_train_set.csv",
    "MIDI_FILES_PATH": "midi_files/",
    "BATCH_SIZE": 8,
    "SEQ_LEN": 600

}


def extract_midi_vector(midi_file_name):
    try:
        m_file = PrettyMIDI(midi_file_name)
        return convert_to_tensor(m_file.get_pitch_class_histogram(), dtype = np.float32)
    except (KeySignatureError, OSError, EOFError, ValueError):
        return None



def read_lyrics_data(path):
    def concat_name(file_names):
        artist = "_".join(file_names[0].split())
        title = "_".join(file_names[1].split())
        file_name = f"{params['MIDI_FILES_PATH']}{artist}_-_{title}.mid"
        return file_name

    data = pd.read_csv(path, sep='\n', header=None)
    data = data.iloc[:, 0].str.rstrip(r"&, ").str.extract(r"([^,]+),([^,]+),(.+)")
    data.columns = ["artist", "title", "lyrics"]
    data["file_name"] = data[["artist", "title"]].apply(concat_name, axis=1)

    return data


def upload_w2v():
    model = gensim.downloader.load('glove-wiki-gigaword-300')
    model.save("word2vec.wordvectors")
    return model

def train_word2vec(sentences):
    model = Word2Vec(sentences=sentences, size=300, window=3, min_count=1, workers=4)
    word_vectors = model.wv
    word_vectors.save("word2vec.wordvectors")


def load_model(path="word2vec.wordvectors"):
    model = KeyedVectors.load(path, mmap='r')
    return model


def get_word_embed(word, model):
    try:
        if word =='<PAD>' or '<UNK>':
            vector = np.zeros((300,))
        else:
            vector = model.wv[word]
        return vector
    except KeyError:
        return None

def save_pickle(file_name, data):
    outfile = open(file_name,'wb')
    pickle.dump(data,outfile)
    outfile.close()

def load_pickle(file_name):
    infile = open(file_name, 'rb')
    data = pickle.load(infile)
    infile.close()

#
# def create_rnn(input_dim, output_dim):
#     model = Sequential()
#     model.add(Embedding(input_dim=output_dim, output_dim=input_dim))
#     model.add(Dropout(0.1))
#     model.add(LSTM(256))
#     model.add(Dense(output_dim, activation = "softmax"))
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
#
#     return model


def create_rnn(units, input_dim, output_size):
    lstm_layer = RNN(LSTMCell(units), input_shape=(None, input_dim), return_sequences=True)
    # lstm_layer = Bidirectional(LSTM(units, input_shape=(None, input_dim * 2), return_sequences=True))

    model = Sequential(
        [
            lstm_layer,
            BatchNormalization(),
            TimeDistributed(Dense(output_size, activation = "softmax")),
        ]
    )
    model.compile(
        # loss='categorical_crossentropy',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model