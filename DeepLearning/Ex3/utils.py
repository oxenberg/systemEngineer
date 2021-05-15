import gensim.downloader
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional, BatchNormalization, TimeDistributed, RNN, LSTMCell, concatenate
from tensorflow import keras
from tensorflow import convert_to_tensor, concat
from pretty_midi import PrettyMIDI
from tensorflow.keras.optimizers import Adam
from mido.midifiles.meta import KeySignatureError
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from tensorflow.keras.preprocessing.text import Tokenizer

params = {
    "TEST_FILE": "lyrics_test_set.csv",
    "TRAIN_FILE": "lyrics_train_set.csv",
    "MIDI_FILES_PATH": "midi_files/",
    "IMAGES_PATH": "images",
    "IMAGE_SAHPE" : (128, 128),
    "CREATE_IMAGES" : False,
    "BATCH_SIZE_AUTO" : 8,
    "BATCH_SIZE": 8,
    "SEQ_LEN": 50,
    "UNITS": 256,
    "MAX_LENGTH": 100,
    "EMBEDDINGS_DIM": 300,
    "MIDI_VECTOR_DIM": 12,
    "MIDI_IMAGE_DIM": 1024,
    "OOV_TOKEN": '<UNK>',
    "PAD_TYPE": 'post',
    "TRUNC_TYPE": 'post'
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


def create_vocab(data):
    words = set()
    for i in data["lyrics"]:
        lyrics = i
        words.update(lyrics)
    words.add('<END>')
    words.add('<PAD>')
    words.add('<UNK>')
    word2index = {w: i for i, w in enumerate(list(words)) if len(w)>0}
    return word2index


def tokenize_songs(sentences, maxlen = 400):
  # Tokenize our training data
  tokenizer = Tokenizer(oov_token=params["OOV_TOKEN"], filters='!"#$%()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
  tokenizer.fit_on_texts(sentences)

  # Get our training data word index
  word_index = tokenizer.word_index

  # Encode training data sentences into sequences
  train_sequences = tokenizer.texts_to_sequences(sentences)

  # Pad the training sequences
  train_padded = pad_sequences(train_sequences, padding=params["PAD_TYPE"], truncating=params["TRUNC_TYPE"], maxlen=maxlen)

  return train_padded, tokenizer, word_index


def create_word_embbedings(word_index,embedding_model, vocab_size):
  embedding_weights = np.zeros((vocab_size, params["EMBEDDINGS_DIM"]))
  for word, index in word_index.items():
      try:
          embedding_weights[index, :] = embedding_model.wv[word]
      except KeyError:
          pass
  return embedding_weights


def create_song_embbedings(data, test):
    embedding_weights_songs = np.zeros((len(data) + len(test), params["MIDI_VECTOR_DIM"]))
    for index, row in data.iterrows():
        try:
            embedding_weights_songs[index, :] = np.array(row["midi_vectors"])
        except KeyError:
            pass

    for _, row in test.iterrows():
        index += 1
        embedding_weights_songs[index, :] = np.array(row["midi_vectors"])

    return embedding_weights_songs


def create_word_features(train_padded):
    x = []
    y = []
    for sentence in train_padded:
        x_ = []
        y_ = []
        for word_index_loc in range(0,len(sentence)-1):
          x_.append(sentence[word_index_loc])
          y_.append(sentence[word_index_loc + 1])
        x.append(x_)
        y.append(y_)
    x = np.array(x)
    y = np.array(y)

    return x, y


def create_song_features(x):
    x_song = []
    for index, value in enumerate(x):
        x_song.append(np.array([index]* len(value)))
    x_song = np.array(x_song)

    return x_song


def create_rnn(units, input_dim, output_size, embedding_weights, midi_dim, midi_weights):
    input_word = keras.Input(shape=(None,), name="word")
    input_song = keras.Input(shape=(None,), name="song")

    #TODO - notice that the two embeddings we use here are the word embeddings, and not the midi vectors
    word_emb = Embedding(output_size, params["EMBEDDINGS_DIM"],
                         embeddings_initializer=keras.initializers.Constant(embedding_weights),
                         trainable=False, mask_zero=True)(input_word)

    midi_emb = Embedding(output_size, midi_dim,
                         embeddings_initializer=keras.initializers.Constant(midi_weights),
                         trainable=False, mask_zero=True)(input_song)

    all_emb = concatenate([word_emb, midi_emb])

    lstm_layer = Bidirectional(LSTM(units, input_shape=(None, input_dim * 2),
                                    return_sequences=True))(all_emb)
    drop_out = Dropout(0.2)(lstm_layer)

    batch_norm = BatchNormalization()(drop_out)

    priority_pred = Dense(output_size, name="priority", activation="softmax")(batch_norm)

    model = keras.Model(
        inputs=[input_word, input_song],
        outputs=[priority_pred])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def get_embeddings(word, midi_data, embedding_model):
    if word=='<PAD>':
        vector_size = 300+midi_data.shape[-1]
        return np.zeros((vector_size,))
    word_vec = get_word_embed(word, embedding_model)
    if word_vec is None:
        return None
    word_vec = convert_to_tensor(word_vec, dtype = np.float32)
    input = concat([word_vec, midi_data], 0)
    return input


def predict_stohcastic(model, song, sentence, word_index, max_length=0, embedings=np.array([])):
    max_length += 1
    if max_length > params["MAX_LENGTH"] or sentence[-1] == "<PAD>":
        return

    embeding = word_index[sentence[-1]]
    embedings = np.append([embedings], [embeding])
    index_emb = embedings.reshape((1, max_length))
    song_emb = np.array([max_length * [song]])
    predictions = model.predict({"word": index_emb, "song": song_emb})

    word = np.random.choice(["<PAD>"] + list(word_index.keys()), 1,
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

        predicted = predict_stohcastic(model, song, vocab, lyrics_gen, vocab)
        predicted = " ".join(predicted)
        predicted_lyrics.append(predicted)
    data["predicted_lyrics"] = predicted_lyrics

    return data


def generate_song(song_num, tokenizer, test, model, train):
    song = song_num + len(train)
    lyrics_oringinal = tokenizer.texts_to_sequences(test["lyrics"])[0]
    first_token = lyrics_oringinal[0]

    lyrics_gen = [tokenizer.index_word[first_token]]

    predict_stohcastic(model, song, lyrics_gen)

    return lyrics_gen