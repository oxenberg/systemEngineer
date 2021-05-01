import pandas as pd

params = {
    "TEST_FILE": "lyrics_test_set.csv",
    "TRAIN_FILE": "lyrics_train_set.csv",
    "MIDI_FILES_PATH": "midi_files/"
}


def read_lyrics_data(path):
    def concat_name(file_names):
        artist = "_".join(file_names[0].split())
        title = "_".join(file_names[1].split())
        file_name = f"{artist}_-_{title}.mid"
        return file_name

    data = pd.read_csv(path, sep='\n', header=None)
    data = data.iloc[:, 0].str.rstrip(r"&, ").str.extract(r"([^,]+),([^,]+),(.+)")
    data.columns = ["artist", "title", "lyrics"]
    data["file_name"] = data[["artist", "title"]].apply(concat_name, axis=1)

    return data
