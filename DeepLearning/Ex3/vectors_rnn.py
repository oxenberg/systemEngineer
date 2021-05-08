from utils import *
from matplotlib import pyplot as plt
from image_rnn import *
from tqdm import tqdm

from glob import glob

# print("uploading model")
# model = upload_w2v()

# print("getting vector")
# v = get_word_embed("all", model)
# print(v.shape)


def create_vocab(data):
    words = set()
    # lengths = []
    for i in data["lyrics"]:
        lyrics = i
        words.update(lyrics)
    words.add('<END>')
    words.add('<PAD>')
    words.add('<UNK>')
    word2index = {w: i for i, w in enumerate(list(words)) if len(w)>0}
    # lengths2 = [num for num in lengths if num>600]
    # print(len(lengths2))
    return word2index


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

# def convert_y(y, vocab_size):
#     y = np.array(y)
#     y_train = np.zeros((y.size, vocab_size))
#     y_train[np.arange(y.size), y] = 1
#     return y_train

def preprocess_data(train_data, midi_func, embedding_model, input_dim, output_dim, vocab, seq_len=params["SEQ_LEN"]):
    song_counter = 0
    x_train = np.zeros((len(train_data), seq_len,input_dim))
    y_train = np.zeros((len(train_data), seq_len))
    for i, row in train_data.iterrows():
        song_lyrics = row["lyrics"].split(" ")
        if len(song_lyrics)<seq_len:
            padding_size = seq_len - len(song_lyrics)
            song_lyrics = song_lyrics + padding_size*['<PAD>']
        elif len(song_lyrics)>=seq_len:
            song_lyrics = song_lyrics[:seq_len]
        midi_data_for_model = row["midi_vectors"]
        if midi_data_for_model is None:
            continue
        word_counter = 0
        words_labels = []
        for j in range(len(song_lyrics)-1):
            word = song_lyrics[j]
            if word not in vocab:
              word = '<UNK>'
            next_word = song_lyrics[j+1]
            if next_word in vocab:
              next_word_index = vocab[next_word]
            else:
              next_word_index = vocab['<UNK>']
            input_vec = get_embeddings(word, midi_data_for_model, embedding_model)
            if input_vec is None:
                continue
            x_train[song_counter][word_counter] = input_vec
            words_labels.append(next_word_index)
            word_counter+=1
        words_labels.append(vocab['<END>'])
        y_train[song_counter] = np.array(words_labels)
        song_counter+=1

    return x_train, y_train

def filter_data(train_data,midi_func, seq_len=params["SEQ_LEN"]):
    midi_vectors = []
    # to_many_words = []
    for i, row in train_data.iterrows():
        # song_lyrics = row["lyrics"].split(" ")
        # if len(song_lyrics) > seq_len:
        #     to_many_words.append(True)
        # else:
        #     to_many_words.append(False)
        midi_data_for_model = midi_func(row['file_name'])
        midi_vectors.append(midi_data_for_model)
    # train_data['seq'] = to_many_words
    train_data['midi_vectors'] = midi_vectors

    # data = train_data[train_data['seq']==False]
    data = train_data[train_data['midi_vectors'].notnull()].reset_index(drop=True)
    return data

# #
# data = read_lyrics_data(params["TRAIN_FILE"])
# test = read_lyrics_data(params["TEST_FILE"])
# #
# # sentences = data['lyrics']
# # # sentences = [sentence.split(" ") for sentence in sentences]
# # train_word2vec(sentences)
# filename = 'train_with_midi_vectors'
# file_name_test = 'test_with_midi_vectors'
# #
# # autoencoder = Autoencoder(params['IMAGES_PATH'], params['IMAGE_SAHPE'], epochs=10, batch=params['BATCH_SIZE_AUTO'])
# # autoencoder.fit()
#
# data = filter_data(data, extract_midi_vector)
# test = filter_data(test, extract_midi_vector)
# # #
# save_pickle(filename, data)
# save_pickle(file_name_test, test)
# # data = load_pickle(filename)
# # test = load_pickle(file_name_test)
# #
# # # # data = data.iloc[:1]
# # vocab = create_vocab(data)
# # print(list(vocab.keys()))
# # vocab_size = len(vocab)
# # input_dim = 312
# # units = 256
# # embedding_model = load_model()
#
#
#
# # x_train, y_train = preprocess_data(data, extract_midi_vector, embedding_model, input_dim, vocab_size, vocab)
# # x_test, y_test = preprocess_data(test, extract_midi_vector, embedding_model, input_dim, vocab_size, vocab)
# # print(x_train.shape)
# # model = create_rnn(params["UNITS"], input_dim, vocab_size)
# # print("fitting model")
# # model.fit(
# #     x_train, y_train, batch_size=params["BATCH_SIZE"], epochs=5
# # )
#

def create_images():
    images_path = []
    files = pd.read_csv("diff.txt", header=None, names=["song_name"])
    for song_name in tqdm(files["song_name"]):
        file = f"{params['MIDI_FILES_PATH']}{song_name}.mid"
        try:
            pm = pretty_midi.PrettyMIDI(file)
        except:
            print("could not read midi files")
        try:
            image_path = plot_piano_roll(pm, song_name, folder="images")
            images_path.append(image_path)
        except:
            print("could not create plot")
            images_path.append(None)


create_images()