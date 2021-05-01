from utils import *
from glob import glob

# print("uploading model")
# model = upload_w2v()

# print("getting vector")
# v = get_word_embed("all", model)
# print(v.shape)




def create_vocab(data):
    words = set()
    for i in data["lyrics"]:
        lyrics = i.lower().split(" ")
        words.update(lyrics)
    words.add('<END>')
    word2index = {w: i for i, w in enumerate(list(words))}

    return word2index

def get_embeddings(word, midi_data, embedding_model):
    word_vec = get_word_embed(word, embedding_model)
    if word_vec is None:
        return None
    word_vec = convert_to_tensor(word_vec, dtype = np.float32)
    input = concat([word_vec, midi_data], 0)
    return input


def train_model(train_data, midi_func, embedding_model, input_dim, output_dim, vocab):
    train_df = pd.DataFrame(columns=["current", "next"])
    for i, row in train_data.iterrows():
        midi_data_for_model = midi_func(row['file_name'])
        if midi_data_for_model is None:
            continue
        song_lyrics = row["lyrics"].split(" ")
        for j in range(len(song_lyrics)-1):
            word = song_lyrics[j]
            next_word_index = vocab[song_lyrics[j+1]]
            input_vec = get_embeddings(word, midi_data_for_model, embedding_model)
            if input_vec is None:
                continue
            train_df.append([[input_vec, next_word_index]])
        input_vec = get_embeddings(song_lyrics[-1], midi_data_for_model, embedding_model)
        train_df.append([[input_vec, vocab['<END>']]])

    X = train_df[:, :-1]
    y = train_df[:, -1]
    # y = tensorflow.keras.utils.to_categorical(train_df[:, -1], num_classes=len(vocab))
    lstm = create_rnn(input_dim, output_dim)

    lstm.fit(X, y, batch_size = params["BATCH_SIZE"], epochs = 1, validation_split = 0.25)


data = read_lyrics_data(params["TRAIN_FILE"])
vocab = create_vocab(data)
vocab_size = len(vocab)
input_dim = (312, )
embedding_model = load_model()

train_model(data, extract_midi_vector, embedding_model, input_dim, vocab_size, vocab)