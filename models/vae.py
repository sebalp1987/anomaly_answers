import pickle
import pandas as pd
import numpy as np
from keras import Input, layers, backend as K, objectives
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from models.ae import DeepAutoencoder

latent_dim = 2

def sampling(args):
    """
    This is going to create the latent space, generating a normal probability distribution from the input data
    """
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)

    return z_mean + K.exp(z_log_var / 2) * epsilon

class VariationalAutoencoder(object):
    def __init__(self, batch_size, latent_dim):
        self.batch_size = batch_size
        self.latent_dim = latent_dim

    def vae_loss(self, x, x_decoded_mean, args):
        """
        We train with two loss functions. Reconstruction loss force decoded samples to match to X (just like
        autoencoder). KL loss (latent loss) calculates the divergence between the learned latent distribution
        derived from z_mean and z_logsigma and the original distribution of X.
        """
        z_mean, z_logsigma = args
        reconstruction_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        latent_loss = -0.5 * K.mean(1 + z_logsigma - K.square(z_mean) - K.exp(z_logsigma), axis=-1)
        print(reconstruction_loss)
        print(latent_loss)
        return K.mean(reconstruction_loss + latent_loss)


if __name__ == '__main__':
    import time
    import os

    seed = 42
    np.random.seed(seed)

    os.chdir("U:\\5FINDER\\resources\\test_data")

    pickle_off = open("Train.pickle", "rb")
    train = pickle.load(pickle_off)
    pickle_off = open("Valid.pickle", "rb")
    valid = pickle.load(pickle_off)
    '''
    json_off = open('Test.ttraint', 'r', encoding='utf8')
    train = json.load(json_off)
    '''
    columns = train[0]
    df = pd.DataFrame(train[1], columns=columns)
    del df['p_churn']
    del df['p_id']
    for i in df.columns.values.tolist():
        df[i] = df[i].map(float)
        df[i] = (df[i] - df[i].mean()) / df[i].std()
    train = df.values
    cols = train.shape[1]

    valid = pd.DataFrame(valid[1], columns=columns)
    del valid['p_churn']
    del valid['p_id']
    for i in valid.columns.values.tolist():
        valid[i] = valid[i].map(float)
        valid[i] = (valid[i] - valid[i].mean()) / valid[i].std()
    valid = valid.values

    learning_rate = 0.001
    epochs = 10
    batch_size = 5000
    verbose = True
    x = Input(shape=(cols, ))
    ae = DeepAutoencoder(n_cols=cols, activation='relu', prob_dropout=None, dimension_node=1)
    vae = VariationalAutoencoder(batch_size=batch_size, latent_dim=latent_dim)
    encoded, intermediate_dim = ae.encoded(x, change_encode_name='Encoder')
    print('First_Layer', encoded)

    # We generate z_mean and sigma from the encoded distribution
    z_mean = layers.Dense(latent_dim, name='z_mean')(encoded)
    z_log_var = layers.Dense(latent_dim, name='z_var')(encoded)

    # We generate the Distribution of Z (Lambda wraps in a layer an arbitrary function)
    Z = layers.Lambda(sampling, output_shape=(latent_dim, ))(([z_mean, z_log_var]))
    print('layer Z: ', Z)

    # We decode Z
    decode_z = layers.Dense(intermediate_dim, activation='relu', name='decode_Z')(Z)
    print(decode_z)

    # We decode X from Z
    decode_x = layers.Dense(cols, activation='sigmoid', name='decode_xZ')(decode_z)
    print(decode_x)

    # VAE Model
    vae_model = Model(x, decode_x)
    print(vae_model.summary())
    loss = vae.vae_loss(x, decode_x, [z_mean, z_log_var])
    print(loss)
    vae_model.add_loss(loss)
    optimizer = SGD(lr=learning_rate)
    vae_model.compile(optimizer=optimizer)
    print(vae_model.summary())
    early_stopping_monitor = EarlyStopping(patience=2)
    history = vae_model.fit(train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stopping_monitor],
        shuffle=True, validation_data=[valid, None])

    history_dict = history.history
    print(history_dict.keys())

    loss = history_dict['loss']
    print(loss)
    val_loss = history_dict['val_loss']

    # Inputs in the latent space (2 dimensions)
    encoder = Model(x, z_mean)







