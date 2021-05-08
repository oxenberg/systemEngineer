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
import re

params = {
    "TEST_FILE": "lyrics_test_set.csv",
    "TRAIN_FILE": "lyrics_train_set.csv",
    "MIDI_FILES_PATH": "midi_files/",
    "IMAGES_PATH": "images",
    "IMAGE_SAHPE" : (128, 128),
    "CREATE_IMAGES" : True,
    "BATCH_SIZE_AUTO" : 8,
    "BATCH_SIZE": 8,
    "SEQ_LEN": 50,
    "UNITS": 256,
    "MAX_LENGTH" : 100,
    "EMBEDDINGS_DIM": 300,
    "MIDI_VECTOR_DIM": 12,
    "MIDI_IMAGE_DIM": 1024
}


def text_preprocess(text):

    text = text.lower()
    padding_text = re.sub(r'[^\w\s&]', r'', text)
    padding_text = padding_text.split(" ")
    padding_text = [w for w in padding_text if len(w)>0]
    return padding_text


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
    # data["lyrics"] = data["lyrics"].apply(text_preprocess)

    return data


def upload_w2v():
    model = gensim.downloader.load('glove-wiki-gigaword-300')
    model.save("word2vec.wordvectors")


def train_word2vec(sentences):
    model = Word2Vec(sentences=sentences, size=300, window=3, min_count=1, workers=4)
    word_vectors = model.wv
    word_vectors.save("word2vec.wordvectors")


def load_model(path="word2vec.wordvectors"):
    model = KeyedVectors.load(path, mmap='r')
    return model


def get_word_embed(word, model):
    try:
        if word =='<PAD>' or word =='<UNK>':
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
    return data

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


def predict_stohcastic(model, song, sentence, vocab, max_length=0, embedings=np.array([])):
    max_length += 1
    if max_length > params['MAX_LENGTH'] or sentence[-1] == "<END>":
        return " ".join(sentence)

    embeding = get_embeddings(sentence[-1], song, embedding_model)
    embedings = np.append([embedings], [embeding])
    embedings_reshape = embedings.reshape((1, max_length, embeding.shape[-1]))
    predictions = model.predict(embedings_reshape)

    word = np.random.choice(list(vocab.keys()), 1,
                            p=predictions[0][-1])[0]
    sentence.append(word)

    predict_stohcastic(model, song, sentence, max_length, embedings)


def predict_songs(data, model, vocab):
    predicted_lyrics = []
    for i, row in data.iterrows():
        song = row["midi_vectors"]

        original_lyrics = row["lyrics"]
        #TODO: check if lyrics is a list or a string
        lyrics_gen = [original_lyrics[1]]

        predicted_lyrics = predict_stohcastic(model, song, lyrics_gen, vocab)
    data["predicted_lyrics"] = predicted_lyrics

    return data