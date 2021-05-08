from utils import *
from image_rnn import *
from vectors_rnn import *



#TODO: Add the tokenizer to the data and make sure everything runs
def main():
    model = input("Enter one of the followings: \n 1 - running the model with images \n 2 - running the model with melody histogram")

    # Reading train and test data:
    train = read_lyrics_data(params["TRAIN_FILE"])
    test = read_lyrics_data(params["TEST_FILE"])

    #Uploading word2vec pre-trained embeddinggs:
    upload_w2v()
    embedding_model = load_model()

    if model == 1:
        input_dim = params["EMBEDDINGS_DIM"]+ params["MIDI_IMAGE_DIM"]
        autoencoder = Autoencoder(params['IMAGES_PATH'], params['IMAGE_SAHPE'], epochs=5,
                                  batch=params['BATCH_SIZE_AUTO'])
        autoencoder.fit()
        midi_func = autoencoder.create_image_embeddings
    else:
        input_dim = params["EMBEDDINGS_DIM"] + params["MIDI_VECTOR_DIM"]
        midi_func = extract_midi_vector

    # filter data to only include rows with midi files we can open:
    train = filter_data(train, midi_func)
    test = filter_data(test, midi_func)

    sentences = train['lyrics']

    train_padded, tokenizer, word_index = tokenize_songs(sentences)
    vocabulary_size = len(word_index) + 1  # we start the count from 1
    embedding_weights = create_word_embbedings(word_index, embedding_model, vocabulary_size)
    #TODO: How do we know which approach do we use here? How can we know what vectors to use?
    embedding_weights_songs = create_song_embbedings(train, test)

    x, y = create_word_features(train_padded)
    x_song = create_song_features(x)

# TODO : input dim here doesn't make sense.
    model = create_rnn(params["UNITS"], input_dim, vocabulary_size, embedding_weights)

    model.fit(x_train, y_train, batch_size=params["BATCH_SIZE"], epochs=5)

    #TODO: check if we need vocab
    #TODO: We need to add the generate song or created, but it only takes into consideration one song. we need to run this on all test data
    test = predict_songs(test, model, vocab)


if __name__ == "__main__":
    # execute only if run as a script
    main()