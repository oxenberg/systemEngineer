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
from sklearn.preprocessing import MinMaxScaler

# params = {
#     "BATCH_SIZE": 128,
#     "NOISE_DIM": 100,
#     "EXAMPLES_TO_GENERATE": 100,
#     "DENSE_DIM" : 16,
#     "EPOCHS": 500,
#     "CHECKPOINT_PATH": "~/training_checkpoint/",
#     "FILES": ["german_credit.arff"],
#     "BUFFER_SIZE": 1000
# }


params = {
    "BATCH_SIZE": 128,
    "NOISE_DIM": 80,
    "EXAMPLES_TO_GENERATE": 100,
    "DENSE_DIM" : 4,
    "EPOCHS": 1000,
    "CHECKPOINT_PATH": "~/training_checkpoint/",
    "FILES": ["german_credit.arff"],
    "BUFFER_SIZE": 1000
}

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

gen_train_loss = tf.keras.metrics.Mean('gen_train_loss', dtype=tf.float32)
disc_train_loss = tf.keras.metrics.Mean('disc_train_loss', dtype=tf.float32)

# generator_optimizer = Adam(1e-4)
# discriminator_optimizer = Adam(1e-4)

generator_optimizer = Adam(0.0007)
discriminator_optimizer = Adam(0.0007)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def read_data(path):
    data = arff.loadarff(path)
    data = pd.DataFrame(data[0])
    return data


# def build_generator_model(batch_size, input_shape, dense_dim, output_dim):
#     input = Input(shape=input_shape, batch_size=batch_size)
#     x = Dense(dense_dim, activation='relu')(input)
#     x = Dense(dense_dim * 2, activation='relu')(x)
#     x = Dense(dense_dim * 4, activation='relu')(x)
#     x = Dense(output_dim)(x)
#     model = Model(inputs=input, outputs=x)
#     return model

def build_generator_model(batch_size, input_shape, dense_dim, output_dim):
    input = Input(shape=input_shape, batch_size=batch_size)
    x = Dense(dense_dim, activation='relu')(input)
    x = Dense(dense_dim * 2, activation='relu')(x)
    # x = Dense(dense_dim * 4, activation='relu')(x)
    x = Dense(output_dim)(x)
    model = Model(inputs=input, outputs=x)
    return model


# def build_discriminator_model(batch_size, input_shape, dense_dim, output_shape=1):
#     input = Input(shape=input_shape, batch_size=batch_size)
#     x = Dense(dense_dim * 4, activation='relu')(input)
#     x = Dropout(0.1)(x)
#     x = Dense(dense_dim * 2, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(dense_dim, activation='relu')(x)
#     x = Dense(output_shape, activation='sigmoid')(x)
#     model = Model(inputs=input, outputs=x)
#     return model

def build_discriminator_model(batch_size, input_shape, dense_dim, output_shape=1):
    input = Input(shape=input_shape, batch_size=batch_size)
    # x = Dense(dense_dim * 4, activation='relu')(input)
    # x = Dropout(0.1)(x)
    x = Dense(dense_dim * 2, activation='relu')(input)
    # x = Dropout(0.1)(x)
    x = Dense(dense_dim, activation='relu')(x)
    x = Dense(output_shape, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=x)
    return model


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# TODO do we need to generate the predictions here?
def train_model(data, generator, discriminator, checkpoint, test_noise):
    for epoch in tqdm(range(params["EPOCHS"])):
        start = time.time()

        for batch in data:
            step(batch, generator, discriminator)
        with train_summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_train_loss.result(), step=epoch)
            tf.summary.scalar('disc_loss', disc_train_loss.result(), step=epoch)

        generate_samples(generator, test_noise)
        # # Save the model every epoch
        # checkpoint.save(file_prefix=params["CHECKPOINT_PATH"])

        # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        gen_train_loss.reset_states()
        disc_train_loss.reset_states()
        # Generate after the final epoch
    generate_samples(generator, test_noise)


def generate_samples(model, test_input):
    predictions = model(test_input, training=False)
    return predictions


@tf.function
def step(samples, generator, discriminator):
    noise = tf.random.normal([params["BATCH_SIZE"], params["NOISE_DIM"]])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_sample = generator(noise, training=True)

        real_output = discriminator(samples, training=True)
        fake_output = discriminator(generated_sample, training=True)

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


def run_GAN(data, num_features):
    output_dim = num_features
    test_noise = tf.random.normal([params["EXAMPLES_TO_GENERATE"], params["NOISE_DIM"]])
    # TODO - change dense dim
    generator = build_generator_model(params["BATCH_SIZE"], params["NOISE_DIM"], params["DENSE_DIM"], output_dim)
    discriminator = build_discriminator_model(params["BATCH_SIZE"], output_dim, params["DENSE_DIM"])
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    train_model(data, generator, discriminator, checkpoint, test_noise)

    analyze_model(generator, discriminator, test_noise, data, num_features)


def analyze_model(generator, discriminator, test_noise, data, num_features):
    samples = generate_samples(generator, test_noise)
    predictions = tf.round(discriminator(samples, training=False))
    passed_as_real = np.array(samples)[np.where(predictions == 1)[0]]
    passed_as_fake = np.array(samples)[np.where(predictions == 0)[0]]
    print(f"{len(passed_as_real)} samples from 100 passed as real samples")
    analyse_passed_as_real(passed_as_real, passed_as_fake, data, num_features)
    return predictions


# TODO: need to create a graph like https://github.com/ydataai/ydata-synthetic/blob/master/examples/regular/gan_example.ipynb
# TODO: with real data points and fake data points
def analyse_passed_as_real(passed_as_real, passed_as_fake, data, num_features):
    columns_index = [0, 1]
    df = tfds.as_dataframe(data.unbatch())
    df = pd.DataFrame(df[""].to_list(), columns=list(np.arange(num_features)))  # columns list to multi columns
    df_to_plot = df.loc[:, columns_index]

    plt.scatter(df[columns_index[0]].values, df[columns_index[1]].values, marker='^', label="real sample")
    plt.scatter(passed_as_real[:, columns_index[0]], passed_as_real[:, columns_index[1]], marker='o',
                label="pass as real")
    plt.scatter(passed_as_fake[:, columns_index[0]], passed_as_fake[:, columns_index[1]], marker='o',
                label="pass as fake")
    plt.xlabel(str(columns_index[0]))
    plt.ylabel(str(columns_index[1]))
    plt.legend(loc='upper left')
    plt.show()

    return



def normalize_data(data, features):
    min_max = MinMaxScaler()
    x = data[features].values  # returns a numpy array
    x_scaled = min_max.fit_transform(x)
    data[features] = pd.DataFrame(x_scaled)
    return data


def prepare_data(file_path):
    df = read_data(file_path)
    df["class"] = df["class"].apply(lambda x: 1 if x=="b'tested_negative'" else 0)
    columns = list(df.columns)
    features = columns[:-1]
    target = columns[-1]
    df = normalize_data(df, features)
    dataset = (tf.data.Dataset.from_tensor_slices(
        (tf.cast(df[features].values, tf.float32))))
    dataset = dataset.shuffle(params["BUFFER_SIZE"]).batch(params["BATCH_SIZE"])
    return dataset, len(features)



def main():
    for file in params["FILES"]:

        if file == "german_credit.arff":
            data, num_features = preprocess_german_df(file)
        else:
            data, num_features = prepare_data(file)


        run_GAN(data, num_features)


if __name__ == "__main__":
    main()
