import glob

images = glob.glob('images/train/*.png')
midi_files = glob.glob('midi_files/*.mid')

def remove_end(data):
    data = [item.split(".")[0].split("\\")[1].lower() for item in data]
    return data

images = remove_end(images)
midi_files = remove_end(midi_files)

diff = set(midi_files) - set(images)
diff = list(diff)

with open("diff.txt","w") as file:
    for i in diff:
        file.write(f"{i}\n")