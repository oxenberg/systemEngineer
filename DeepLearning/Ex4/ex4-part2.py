import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from scipy.io import arff
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

params = {
    "BATCH_SIZE": 128,
    "NOISE_DIM": 100,
    "EXAMPLES_TO_GENERATE": 100,
    "DENSE_DIM" : 32,
    "EPOCHS": 500,
    "MAX_DEPTH": 5, # 3 for diabetes
    "CHECKPOINT_PATH": "~/training_checkpoint/",
    "FILES": ["german_credit.arff"],
    "BUFFER_SIZE": 1000
}

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

gen_train_loss = tf.keras.metrics.Mean('gen_train_loss', dtype=tf.float32)
disc_train_loss = tf.keras.metrics.Mean('disc_train_loss', dtype=tf.float32)

generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def normalize_data(data, features):
    min_max = MinMaxScaler(feature_range=(-1,1))
    x = data[features].values  # returns a numpy array
    x_scaled = min_max.fit_transform(x)
    data[features] = pd.DataFrame(x_scaled)
    return data


def normalize_categorial_data(data, features):
    label_encoder = LabelEncoder()
    data[features] = data[features].apply(label_encoder.fit_transform)
    data[features] = normalize_data(data[features], features)

    return data


def read_data(path):
    data = arff.loadarff(path)
    data = pd.DataFrame(data[0])
    return data

def prepare_data(file_path):
    df = read_data(file_path)
    df["class"] = df["class"].apply(lambda x: 0 if x == str.encode("tested_negative") else 1)
    columns = list(df.columns)
    features = columns[:-1]
    df = normalize_data(df, features)
    return df

def preprocess_german_df(file_path):
    df = read_data(file_path)
    columns = list(df.columns)
    numerical_features  = [2, 5, 8, 11, 13, 18]
    categorial_features = list(set(np.arange(1,len(columns)+1)) - set(numerical_features))
    numerical_features = list(map(str,numerical_features))
    categorial_features = list(map(str,categorial_features))
    df = normalize_data(df, numerical_features)
    df = normalize_categorial_data(df, categorial_features)
    return df


def train_random_forest(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(max_depth=params["MAX_DEPTH"], random_state=0)
    clf.fit(X_train, y_train)
    check_clf_performance(clf, X_test, y_test)

def check_clf_performance(clf, X_test, y_test):
    score = clf.score(X_test, y_test)
    print(f"Accuracy of the classifier:{score}")
    predictions = clf.predict_proba(X_test)
    predictions = predictions[:, 1]
    print(f"Confidence score distribution summary:\n Min value:{min(predictions)}\n"
          f" Max value:{max(predictions)}\n Average value:{np.mean(predictions)}\n")
    plt.hist(predictions)
    plt.show()





def main():
    for file in params["FILES"]:

        if file == "german_credit.arff":
            data = preprocess_german_df(file)
        else:
            data = prepare_data(file)

        train_random_forest(data)
        # run_GAN(data, num_features)



if __name__ == "__main__":
    main()