import pandas as pd
import glob
import pretty_midi
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
from mido.midifiles.meta import KeySignatureError
# Kears imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# our lib
from utils import read_lyrics_data, params

# IMAGES_PATH = "images/train/"
# CREATE_IMAGES = False
# # autoencoder params
# IMAGE_SAHPE = (252, 252)
# BATCH_SIZE_AUTO = 8


def plot_piano_roll(pm, name=None, folder=None):
    ax = plt.axes()
    image_path = f"{params['IMAGES_PATH']}/{folder}/{name}.png"
    # Use librosa's specshow function for displaying the piano roll
    fig = librosa.display.specshow(pm.get_piano_roll(), ax=ax,
                                   hop_length=1, x_axis='time', y_axis='cqt_note')
    ax.axis('off')
    ax.figure.savefig(fname=image_path, bbox_inches='tight', pad_inches=0)
    return


def create_images(data, name):
    print(f"---- create {name} in images file ----")
    images_path = []
    if params['CREATE_IMAGES']:
        for file in tqdm(data["file_name"]):
            try:
                song_name = file.split('/')[1].split(".")[0]
                pm = pretty_midi.PrettyMIDI(file)
                image_path = plot_piano_roll(pm, song_name, folder=name)
                images_path.append(image_path)
            except (KeySignatureError, OSError, EOFError, ValueError) as e:
                print(e)
                # print(f"Exception: could not create {file}")
                images_path.append(None)
        data["image_path"] = images_path
        print(f"save {name} file")
        data.to_csv(f"{name}.csv")


class Autoencoder:

    def __init__(self, images_path, image_shape, epochs, batch, create_images=True):
        self.image_shape_generator = image_shape
        self.image_shape_model = (image_shape[0], image_shape[1], 3)
        self.epochs = epochs
        self.batch = batch

        self.all_data = self.build_data_set(images_path)
        self.generator = self.create_generators()

        self.embedings_model = ""
        self.autoencoder_model = ""

    def build_data_set(self, images_path):
        all_images_name = glob.glob(f"{images_path}/*/*.png")
        all_data = pd.DataFrame(all_images_name, columns=["image_path"])
        all_data["name"] = all_data["image_path"].apply(lambda x: x.split("/")[2].split(".")[0])

        return all_data

    def create_generators(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_gen = train_datagen.flow_from_dataframe(self.all_data, x_col="image_path", seed=42,
                                                      target_size=self.image_shape_generator,
                                                      class_mode="input", shuffle=False, batch_size=self.batch)

        return train_gen

    def build_auto_encoder(self, encoding_dim=32):
        input_img = keras.Input(shape=self.image_shape_model)

        x = layers.Conv2D(16, (6, 6), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((4, 4), padding='same')(x)
        x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = layers.Conv2D(4, (6, 6), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((4, 4))(x)
        x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = keras.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.summary()

        return autoencoder, encoded, input_img

    def fit(self):
        autoencoder, encoder, input_img = self.build_auto_encoder()
        autoencoder.fit(self.generator,
                        epochs=self.epochs,
                        batch_size=self.batch,
                        shuffle=True)

        self.embedings_model = keras.Model(input_img, encoder)
        self.autoencoder_model = autoencoder

    def check_if_exist(self, name):
        '''
        get series and value
        '''
        return self.all_data["name"] == name

    def create_image_embeddings(self, name):
        '''
        this function is the genric function for the encoding part
        need to send to be concated with the word embddibngs
        '''

        name = name.split("/")[1].split(".")[0]
        song = self.all_data[self.all_data["name"] == name]

        # no song exist
        if len(song) == 0:
            return None

        song_path = song["image_path"].values[0]
        image = keras.preprocessing.image.load_img(song_path, target_size=self.image_shape_generator)
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        embeding = self.embedings_model.predict(input_arr).flatten()
        return embeding

    def show_reconstract_exemple_images(self):
        decoded_imgs = self.autoencoder_model.predict(self.generator)
        x_test = self.generator.next()[0]
        n = 2
        plt.figure(figsize=(20, 10))
        for i in range(1, n + 1):
            # Display original
            ax = plt.subplot(2, n, i)
            plt.imshow(x_test[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.imshow(decoded_imgs[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":

    if params['CREATE_IMAGES']:
        # read data from the test train files (not the mid files)
        train = read_lyrics_data(params["TRAIN_FILE"])
        test = read_lyrics_data(params["TEST_FILE"])

        # export piano notes spectrogram to png images in images file
        create_images(train, name="train")
        create_images(test, name="test")

    # autoencoder = Autoencoder(params['IMAGES_PATH'], params['IMAGE_SAHPE'], epochs=1, batch=params['BATCH_SIZE_AUTO'])
    # autoencoder.fit()
    #
    # embeding = autoencoder.create_image_embeddings("midi_files/elmore_james_-_dust_my_broom.mid")
    # print(embeding.shape)
    # autoencoder.show_reconstract_exemple_images()
