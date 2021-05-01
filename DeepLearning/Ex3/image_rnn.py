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
    image_path = f"{IMAGES_PATH}{name}.png"
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
        for file in tqdm(data["file_name"]):
            try:
                name = file.split(".")[0]
                pm = pretty_midi.PrettyMIDI(params["MIDI_FILES_PATH"] + file)
                image_path = plot_piano_roll(pm, name)
                images_path.append(image_path)
            except KeySignatureError:
                print(f"Exception: could not create {file}")
                images_path.append(None)
        data["image_path"] = images_path
        print(f"save {name} file")
        data.to_csv(f"{name}.csv")

# main

# read data from the test train files (not the mid files)
train = read_lyrics_data(params["TRAIN_FILE"])
test = read_lyrics_data(params["TEST_FILE"])

# export piano notes spectrogram to png images in images file
create_images(train, name="train")
create_images(test, name="test")
