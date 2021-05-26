import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,VarLenSparseFeat,get_feature_names
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import glob
import numpy as np
from tqdm import tqdm

delete_coulmns = ["gmt_offset","user_id_hash"]
features_list = ["target_item_taxonomy"]
sparse_features  = ["target_id_hash","syndicator_id_hash","campaign_id_hash","placement_id_hash","publisher_id_hash","source_id_hash","source_item_type","browser_platform","country_code","region"]
target = ['is_click']
multi_value = ["target_item_taxonomy"]
user_data = ["user_id_hash","user_target_recs","user_recs","user_clicks","country_code","region","browser_platform","os_family","day_of_week","time_of_day"]
item_data = ["target_id_hash", "syndicator_id_hash","campaign_id_hash","empiric_calibrated_recs","empiric_clicks","target_item_taxonomy","placement_id_hash","page_view_start_time","publisher_id_hash","source_id_hash","source_item_type"]
key2index = {}
GLOBAL_PATH_TRAIN = "train_data/*/*"
TEST_PATH = "test_file.csv"

def split(x):
    key_ans = x.split('~')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))

# ------------------ main ------------------ #

all_paths = glob.glob(GLOBAL_PATH_TRAIN)

data = pd.DataFrame()
for path in tqdm(all_paths):
    data = data.append(pd.read_csv(path))

test = pd.read_csv(TEST_PATH)
data["Id"] = -1
test["is_click"] = -1
data = pd.concat([test,data])

del test

dense_features  = list(set(data.columns) - set(target) - set(sparse_features) - set(multi_value) - set(delete_coulmns) - set(["Id"]))
data = data.drop(delete_coulmns, axis = 1)

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])

genres_list = list(map(split, data[multi_value[0]].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
# Notice : padding=`post`
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]

varlen_feature_columns = [VarLenSparseFeat(SparseFeat(multi_value[0],vocabulary_size= len(
            key2index) + 1,embedding_dim=4), maxlen=max_len, combiner='mean',weight_name=None)]  # Notice : value 0 is for padding for sequence input feature
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train = data[data["Id"] == -1]
test = data[data["is_click"] == -1]
# train, test = train_test_split(data, test_size=0.2)


train_model_input = {name:train[name].values for name in feature_names}
train_model_input[multi_value[0]] = genres_list[train.index]
test_model_input = {name:test[name].values for name in feature_names}
test_model_input[multi_value[0]] = genres_list[test.index]

target_values = train[target].values
del train
del test
model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary')
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])

history = model.fit(train_model_input, target_values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)

