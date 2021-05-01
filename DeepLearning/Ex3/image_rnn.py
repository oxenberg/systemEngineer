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


def plot_piano_roll(pm, name=None):
    ax = plt.axes()

    # Use librosa's specshow function for displaying the piano roll
    fig = librosa.display.specshow(pm.get_piano_roll(), ax=ax,
                                   hop_length=1, x_axis='time', y_axis='cqt_note')
    ax.axis('off')
    fig.savefig(fname=f"{IMAGES_PATH}{name}.png", bbox_inches='tight', pad_inches=0)


def create_images(data):
    print(f"---- create {data} in images file ----")
    if CREATE_IMAGES:
        for file in tqdm(data["file_name"]):
            try:
                name = file.split(".")[0]
                pm = pretty_midi.PrettyMIDI(params["MIDI_FILES_PATH"] + file)
                plot_piano_roll(pm, name)
            except KeySignatureError:
                print(f"Exception: could not create {file}")


# main

# read data from the test train files (not the mid files)
train = read_lyrics_data(params["TRAIN_FILE"])
test = read_lyrics_data(params["TEST_FILE"])

# export piano notes spectrogram to png images in images file
create_images(train)
create_images(test)
