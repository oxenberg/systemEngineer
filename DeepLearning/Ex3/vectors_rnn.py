from utils import *

print("uploading model")
# model = upload_w2v()
model = load_model()
print("getting vector")
v = get_word_embed("all", model)
print(v)