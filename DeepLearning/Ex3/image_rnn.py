import pandas as pd
import glob
import pretty_midi
import matplotlib.pyplot as plt
import librosa.display
from .utils import read_lyrics_data
TEST_FILE = "lyrics_test_set.csv"
TRAIN_FILE = "lyrics_train_set.csv"
MIDI_FILES_PATH = "midi_files/"

def plot_piano_roll(pm):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(),
                             hop_length=1, x_axis='time', y_axis='cqt_note')
    plt.show()






train = read_lyrics_data(TRAIN_FILE)
test = read_lyrics_data(TEST_FILE)

for file in train["file_name"]:
    pm = pretty_midi.PrettyMIDI(MIDI_FILES_PATH+file)
    plot_piano_roll(pm)