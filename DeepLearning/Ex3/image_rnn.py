import pandas as pd
import glob
import pretty_midi
import matplotlib.pyplot as plt
import librosa.display
from utils import read_lyrics_data, params
from tqdm import tqdm
from mido.midifiles.meta import KeySignatureError





IMAGES_PATH = "images/"
CREATE_IMAGES = True
TRAIN_POSTPROCESS = "train.csv"
TEST_POSTPROCESS = "test.csv"
# autoencoder params
BATCH_SIZE_AUTO = 32


def plot_piano_roll(pm, name=None,folder=None):
    ax = plt.axes()
    image_path = f"{IMAGES_PATH}{folder}/{name}.png"
    # Use librosa's specshow function for displaying the piano roll
    fig = librosa.display.specshow(pm.get_piano_roll(), ax=ax,
                                   hop_length=1, x_axis='time', y_axis='cqt_note')
    ax.axis('off')
    ax.figure.savefig(fname=image_path, bbox_inches='tight', pad_inches=0)
    return

def create_images(data, name):
    print(f"---- create {name} in images file ----")
    images_path = []
    if CREATE_IMAGES:
        for file in tqdm(data["file_name"][132:]):
            try:
                song_name = file.split(".")[0]
                pm = pretty_midi.PrettyMIDI(params["MIDI_FILES_PATH"] + file)
                image_path = plot_piano_roll(pm, song_name,folder = name)
                images_path.append(image_path)
            except (KeySignatureError, OSError):
                print(f"Exception: could not create {file}")
                images_path.append(None)
        data["image_path"] = images_path
        print(f"save {name} file")
        data.to_csv(f"{name}.csv")

        # TODO add 132 files from begining to the data frame image paths

def create_generators(train,test):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(train, x_col="image_path", seed=42,
                                                  class_mode="input", shuffle=False, batch_size=BATCH_SIZE_AUTO)
    test_gen = test_datagen.flow_from_dataframe(test, x_col="image_path", seed=42,
                                                class_mode="input", shuffle=False, batch_size=BATCH_SIZE_AUTO)

    return train_gen,test_gen




def build_auto_encoder(encoding_dim = 32, image_shape = 784):
    input_img = keras.Input(shape= (image_shape,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    decoded = layers.Dense(784, activation='sigmoid')(encoded)

    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder, encoder

#  ------------- main --------------

if CREATE_IMAGES:

    # read data from the test train files (not the mid files)
    train = read_lyrics_data(params["TRAIN_FILE"])
    test = read_lyrics_data(params["TEST_FILE"])

    # export piano notes spectrogram to png images in images file
    create_images(train, name="train")
    create_images(test, name="test")

# after crating csv with paths name, regex fixes and creating images we can import directly the images and
# generators for the images


# train = pd.read_csv(TRAIN_POSTPROCESS)
# test = pd.read_csv(TEST_POSTPROCESS)
#
# train_gen, test_gen = create_generators(train, test)
#
# autoencoder, encoder = build_auto_encoder()
#
# autoencoder.fit(train_gen,
#                 epochs=50,
#                 batch_size=BATCH_SIZE_AUTO,
#                 shuffle=True)




