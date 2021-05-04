from utils import *
from matplotlib import pyplot as plt
import pickle

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
        lyrics = i.lower().split(" ")
        # lengths.append(len(lyrics))
        words.update(lyrics)
    words.add('<END>')
    words.add('<PAD>')
    word2index = {w: i for i, w in enumerate(list(words))}
    # lengths2 = [num for num in lengths if num>600]
    # print(len(lengths2))
    return word2index

def get_embeddings(word, midi_data, embedding_model):
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

def preprocess_data(train_data, midi_func, embedding_model, input_dim, output_dim, vocab, seq_len=600):
    song_counter = 0
    x_train = np.zeros((len(train_data), seq_len,input_dim))
    y_train = np.zeros((len(train_data), seq_len))
    for i, row in train_data.iterrows():
        song_lyrics = row["lyrics"].split(" ")
        if len(song_lyrics)<seq_len:
            padding_size = seq_len - len(song_lyrics)
            song_lyrics = song_lyrics + padding_size*['<PAD>']
        midi_data_for_model = row["midi_vectors"]
        if midi_data_for_model is None:
            continue
        word_counter = 0
        words_labels = []
        for j in range(len(song_lyrics)-1):
            word = song_lyrics[j]
            next_word_index = vocab[song_lyrics[j+1]]
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

def filter_data(train_data,midi_func, seq_len = 600):
    midi_vectors = []
    to_many_words = []
    for i, row in train_data.iterrows():
        song_lyrics = row["lyrics"].split(" ")
        if len(song_lyrics) > seq_len:
            to_many_words.append(True)
        else:
            to_many_words.append(False)
        midi_data_for_model = midi_func(row['file_name'])
        midi_vectors.append(midi_data_for_model)
    train_data['seq'] = to_many_words
    train_data['midi_vectors'] = midi_vectors

    data = train_data[train_data['seq']==False]
    data = data[data['midi_vectors'].notnull()]
    return data


#
# def train_model(train_data, midi_func, embedding_model, input_dim, output_dim, vocab):
#     train_df = pd.DataFrame(columns=["current", "next"])
#     current = []
#     next = []
#     for i, row in train_data.iterrows():
#         midi_data_for_model = midi_func(row['file_name'])
#         if midi_data_for_model is None:
#             continue
#         song_lyrics = row["lyrics"].split(" ")
#         for j in range(len(song_lyrics)-1):
#             word = song_lyrics[j]
#             next_word_index = vocab[song_lyrics[j+1]]
#             input_vec = get_embeddings(word, midi_data_for_model, embedding_model)
#             if input_vec is None:
#                 continue
#             current.append(input_vec)
#             next.append(next_word_index)
#         input_vec = get_embeddings(song_lyrics[-1], midi_data_for_model, embedding_model)
#         current.append(input_vec)
#         next.append(vocab['<END>'])
#     train_df["next"] = next
#     train_df["current"] = current
#     X = train_df.iloc[:, :-1]
#     y = train_df.iloc[:, -1]
#     # y = tensorflow.keras.utils.to_categorical(train_df[:, -1], num_classes=len(vocab))
#     lstm = create_rnn(input_dim, output_dim)
#
#     lstm.fit(X, y, batch_size = params["BATCH_SIZE"], epochs = 1, validation_split = 0.25)


data = read_lyrics_data(params["TRAIN_FILE"])
sentences = data['lyrics']
sentences = [sentence.split(" ") for sentence in sentences]
# train_word2vec(sentences)
data = filter_data(data, extract_midi_vector)
filename = 'train_with_vectors'
outfile = open(filename,'wb')
pickle.dump(data,outfile)
outfile.close()

# # data = data.iloc[:1]
vocab = create_vocab(data)
vocab_size = len(vocab)
input_dim = 312
units = 64
embedding_model = load_model()


x_train, y_train = preprocess_data(data, extract_midi_vector, embedding_model, input_dim, vocab_size, vocab)
model = create_rnn(units, input_dim, vocab_size)
print("fitting model")
model.fit(
    x_train, y_train, batch_size=params["BATCH_SIZE"], epochs=1
)
# train_model(data, extract_midi_vector, embedding_model, input_dim, vocab_size, vocab)