import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from keras import Input, layers, regularizers
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping


class DeepAutoencoder(object):
    def __init__(self, n_cols, activation, prob_dropout, dimension_node):
        """
        :param n_cols: Number of cols of datastet 
        :param activation: Activation function
        :param prob_dropout: proportion to dropout
        :param dimension_node: Number of depth of encoder/decoder
        """
        self.n_cols = n_cols
        self.activation = activation
        self.prob_dropout = prob_dropout
        self.dimension_node = dimension_node

    def encoded(self, input_layer, sparsity_const=0.0002, change_encode_name=None):
        """
        Generate the encode layers
        :param sparsity_const: Restrict some nodes and not all (as PCA), using regularization strategy
        :param prob_dropout: Probability of droping out
        :return: encode layer
        """
        dim = self.dimension_node
        cols = self.n_cols
        node_size = int(cols / dim)
        nodes_range = range(cols - node_size, node_size - 1, -node_size)

        for nodes in nodes_range:
            last_node = nodes
            print(last_node)
            layer_name = 'Encoded_model_' + str(nodes)
            if change_encode_name is not None:
                layer_name = 'Encoded_model_' + change_encode_name +'_' + str(nodes)

            if sparsity_const is not None:
                input_layer = layers.Dense(nodes, activation=self.activation, name=layer_name, activity_regularizer=
                                            regularizers.l1_l2(l1=sparsity_const, l2=sparsity_const))(input_layer)
            else:
                input_layer = layers.Dense(nodes, activation=self.activation, name=layer_name)(input_layer)
            if self.prob_dropout is not None:
                input_layer = layers.Dropout(self.prob_dropout)(input_layer)

        if dim == 1:
            last_node = int(cols/2)
            if sparsity_const is not None:
                input_layer = layers.Dense(int(cols / 2), activation=self.activation, name='Encode_Layer',
                                           activity_regularizer=regularizers.l1_l2(l1=sparsity_const,
                                                                                   l2=sparsity_const))(input_layer)
            else:
                input_layer = layers.Dense(int(cols / 2), activation=self.activation, name='Encode_Layer')(input_layer)
            if self.prob_dropout is not None:
                input_layer = layers.Dropout(self.prob_dropout)(input_layer)



        encode_layer = input_layer
        print(last_node)
        return encode_layer, last_node

    def decoded(self, encode_layer, change_decode_name=None, final_activation='softmax'):
        """
        Generate the decoded model
        :param nodes_range: (init_nodes, final_nodes + 1, +steps)
        :return: decoded layer
        """
        dim = self.dimension_node
        cols = self.n_cols
        node_size = int(cols / dim)
        nodes_range = range(node_size*2, cols, node_size)
        name = 'Decoded_Model'
        for nodes in nodes_range:
            layer_name = 'Decoded_model_' + str(nodes)
            if change_decode_name is not None:
                layer_name = 'Encoded_model_' + change_decode_name +'_' + str(nodes)
                name = 'Decoded_Model_' + change_decode_name
            encode_layer = layers.Dense(nodes, activation=self.activation, name=layer_name)(encode_layer)

        encode_layer = layers.Dense(self.n_cols, activation=final_activation, name=name)(encode_layer)

        decode_layer = encode_layer
        return decode_layer

    def autoencoder(self, input_tensor):
        """
        Generate the autoencoder model
        :return: autoencoder model
        """
        encode_layer, _ = DeepAutoencoder.encoded(self, input_tensor)
        decode_layer = DeepAutoencoder.decoded(self, encode_layer)
        autoencoder = Model(input_tensor, decode_layer)
        print(autoencoder.summary())

        return autoencoder

    def fit(self, X, X_valid, learning_rate=0.001, loss_function='categorical_crossentropy', epochs=500,
            batch_size=500, verbose=True, callback_list=[]):
        input_tensor = Input(shape=(self.n_cols,), name='Input')
        autoencoder = DeepAutoencoder.autoencoder(self, input_tensor)
        optimizer = SGD(lr=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss=loss_function)
        history = autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callback_list,
                                  shuffle=True, validation_data=X_valid)

def test(input_dim_list):
    start_time = time.time()


if __name__ == '__main__':
    import time
    import os

    seed = 42
    np.random.seed(seed)

    os.chdir("U:\\5FINDER\\resources\\test_data")

    pickle_off = open("Train.pickle", "rb")
    x = pickle.load(pickle_off)
    pickle_off = open("Valid.pickle", "rb")
    valid = pickle.load(pickle_off)
    '''
    json_off = open('Test.txt', 'r', encoding='utf8')
    x = json.load(json_off)
    '''
    columns = x[0]
    df = pd.DataFrame(x[1], columns=columns)
    del df['p_churn']
    del df['p_id']
    for i in df.columns.values.tolist():
        df[i] = df[i].map(float)
        df[i] = (df[i] - df[i].mean()) / df[i].std()
    x = df.values
    cols = x.shape[1]

    valid = pd.DataFrame(valid[1], columns=columns)
    del valid['p_churn']
    del valid['p_id']
    for i in valid.columns.values.tolist():
        valid[i] = valid[i].map(float)
        valid[i] = (valid[i] - valid[i].mean()) / valid[i].std()
    print(valid.shape)
    valid = valid.values

    start_time = time.time()
    ae = DeepAutoencoder(n_cols=cols, activation='relu', prob_dropout=None, dimension_node=3)
    early_stopping_monitor = EarlyStopping(patience=2)
    ae.fit(x, X_valid=[valid, valid], callback_list=[early_stopping_monitor], batch_size=5000, epochs=2,
           learning_rate=0.01)

    # MODELS
    input_tensor = Input(shape=(cols,))
    input_encoded = Input(shape=(41,))

    encoded, _ = ae.encoded(input_tensor)
    decoded = ae.decoded(input_encoded)
    autoencoder = ae.autoencoder(input_tensor)

    # ENCODER MODEL
    encoder = Model(input_tensor, encoded)

    # DECODER MODEL
    decoder = Model(input_encoded, decoded)

    # PREDICT
    encode_valid = encoder.predict(valid)
    decoded_valid = decoder.predict(encode_valid)
    autoencoded_valid = autoencoder.predict(valid)

    # Compare the latent vectors produced from the encoder using the original X versus the output of the autoencoder
    # More approximate to 1 implies similarity
    def cosine_similarity(x, y):
        return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))


    for col in range(valid.shape[0]):
        print(col)
        print(cosine_similarity(valid[col], decoded_valid[col]))

    # Check similarity
    y_vec = encoder.predict(autoencoded_valid)

    for col in range(encode_valid.shape[0]):
        print(col)
        print(cosine_similarity(encode_valid[col], y_vec[col]))


