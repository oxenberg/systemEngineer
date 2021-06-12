import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
# import tensorflow_decision_forests as tfdf

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

tf.executing_eagerly()


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
    dataset = (tf.data.Dataset.from_tensor_slices(
        (tf.cast(df[features].values, tf.float32))))
    dataset = dataset.shuffle(params["BUFFER_SIZE"]).batch(params["BATCH_SIZE"])

    return df, dataset

def preprocess_german_df(file_path):
    df = read_data(file_path)
    columns = list(df.columns)
    numerical_features  = [2, 5, 8, 11, 13, 18]
    categorial_features = list(set(np.arange(1,len(columns)+1)) - set(numerical_features))
    numerical_features = list(map(str,numerical_features))
    categorial_features = list(map(str,categorial_features))
    df = normalize_data(df, numerical_features)
    df = normalize_categorial_data(df, categorial_features)

    dataset = (tf.data.Dataset.from_tensor_slices(
        (tf.cast(df[numerical_features+categorial_features].values, tf.float32))))
    dataset = dataset.shuffle(params["BUFFER_SIZE"]).batch(params["BATCH_SIZE"])
    return df, dataset

def train_random_forest(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(max_depth=params["MAX_DEPTH"], random_state=0)
    clf.fit(X_train, y_train)
    check_clf_performance(clf, X_test, y_test)
    return clf

def check_clf_performance(clf, X_test, y_test):
    score = clf.score(X_test, y_test)
    print(f"Accuracy of the classifier:{score}")
    predictions = clf.predict_proba(X_test)
    predictions = predictions[:, 1]
    print(f"Confidence score distribution summary:\n Min value:{min(predictions)}\n"
          f" Max value:{max(predictions)}\n Average value:{np.mean(predictions)}\n")
    plt.hist(predictions)
    plt.show()


def build_generator_model(batch_size, input_shape_noise, dense_dim, output_dim):
    input1 = Input(shape=input_shape_noise, batch_size=batch_size)
    input2 = Input(shape = 1, batch_size=batch_size)

    x = concatenate([input1, input2])

    x = Dense(dense_dim, activation='relu')(x)
    x = Dense(dense_dim * 2, activation='relu')(x)
    x = Dense(dense_dim * 4, activation='relu')(x)
    x = Dense(output_dim)(x)
    model = Model(inputs=[input1,input2], outputs=x)
    return model

def build_discriminator_model(batch_size, input_shape, dense_dim, output_shape=1):
    input = Input(shape=input_shape, batch_size=batch_size)
    inputC = Input(shape=(1,), batch_size=batch_size)
    inputY = Input(shape=(1,), batch_size=batch_size)

    x = concatenate([input, inputC,inputY])

    x = Dense(dense_dim * 4, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(dense_dim * 2, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(dense_dim, activation='relu')(x)
    x = Dense(output_shape, activation='sigmoid')(x)
    model = Model(inputs=[input,inputC,inputY], outputs=x)
    return model

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss




def step(samples, generator, discriminator, blackOrWhiteBox,C):
    noise = tf.random.normal([params["BATCH_SIZE"], params["NOISE_DIM"]])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_sample = generator([noise , C], training=True)

        y = blackOrWhiteBox.predict_proba(np.array(generated_sample))[:, 1]  # get the proba of y =1
        # generated_sample = generator([noise, C], training=True)

        real_output = []
        fake_output = []
        indexes = np.random.choice(params["BATCH_SIZE"], int(params["BATCH_SIZE"] / 2))
        for index in range(params["BATCH_SIZE"]):

            if index in indexes:
                sample_tuple = (generated_sample[index], y[index], C[index])
                real_output.append(sample_tuple)
            else:
                sample_tuple = (generated_sample[index], C[index], y[index])
                fake_output.append(sample_tuple)

        sample_real, y_real, c_real = zip(*real_output)
        sample_fake, c_fake, y_fake = list(zip(*fake_output))

        real_output = discriminator([sample_real, tf.convert_to_tensor(y_real), tf.convert_to_tensor(c_real)], training=True)
        fake_output = discriminator([sample_fake, tf.convert_to_tensor(c_fake), tf.convert_to_tensor(y_fake)], training=True)


        # real_output = discriminator(generated_sample, training = True)
        # fake_output = discriminator(generated_sample, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_train_loss(gen_loss)
    disc_train_loss(disc_loss)

    # tf.print("discriminator loss:", disc_loss, ",generator loss:", gen_loss)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # disc_tape.watch(disc_loss)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train_model(data, generator, discriminator, checkpoint, test_noise, blackOrWhiteBox):
    for epoch in tqdm(range(params["EPOCHS"])):
        start = time.time()

        for batch in data:
            C = np.random.rand(1,params["BATCH_SIZE"]).T
            step(batch, generator, discriminator, blackOrWhiteBox, C)
        with train_summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_train_loss.result(), step=epoch)
            tf.summary.scalar('disc_loss', disc_train_loss.result(), step=epoch)

        # # Save the model every epoch
        # checkpoint.save(file_prefix=params["CHECKPOINT_PATH"])

        # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        gen_train_loss.reset_states()
        disc_train_loss.reset_states()
        # Generate after the final epoch
    # generate_samples(generator, test_noise)

def run_GAN(data, num_features,blackOrWhiteBox):
    output_dim = num_features
    test_noise = tf.random.normal([params["EXAMPLES_TO_GENERATE"], params["NOISE_DIM"]])
    # TODO - change dense dim
    generator = build_generator_model(params["BATCH_SIZE"], params["NOISE_DIM"], params["DENSE_DIM"], output_dim)
    discriminator = build_discriminator_model(params["BATCH_SIZE"], output_dim, params["DENSE_DIM"])
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    train_model(data, generator, discriminator, checkpoint, test_noise,blackOrWhiteBox)

    # analyze_model(generator, discriminator, test_noise, data, num_features)





def main():

    print("VERSION TF" + tf.__version__)
    print("VERSION NP" + np.__version__)

    for file in params["FILES"]:

        if file == "german_credit.arff":
            data, dataset = preprocess_german_df(file)
        else:
            data, dataset = prepare_data(file)

        blackOrWhiteBox = train_random_forest(data)

        columns_len = data.shape[1] - 1
        run_GAN(dataset, columns_len,blackOrWhiteBox)



if __name__ == "__main__":
    main()