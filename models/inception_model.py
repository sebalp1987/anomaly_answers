import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras import layers, Input, regularizers
from keras.models import Model
from sklearn import metrics
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import pickle


class InceptionModel(object):
    def __init__(self, n_cols, node_size=100, branch_number=4,
                 prob_dropout=0.1, sparsity_const=0.0002, activation='relu'):

        self.n_cols = n_cols
        self.node_size = node_size
        self.branch_number = branch_number
        self.activation = activation
        self.prob_dropout = prob_dropout
        self.sparsity_const = sparsity_const

        input_layer = Input(shape=(n_cols,))

        x = input_layer
        branches = []
        for branch in range(0, branch_number + 1, 1):
            if sparsity_const is not None:
                branch_i = layers.Dense(node_size, activation=self.activation, activity_regularizer=
                regularizers.l1_l2(l1=sparsity_const, l2=sparsity_const))(x)
            else:
                branch_i = layers.Dense(node_size, activation=self.activation, activity_regularizer=
                regularizers.l1_l2(l1=sparsity_const, l2=sparsity_const))(x)
            if self.prob_dropout is not None:
                branch_i = layers.Dropout(self.prob_dropout)(branch_i)

            branches.append(branch_i)

        branches = layers.concatenate(branches, axis=-1)
        output_tensor = layers.Dense(2, activation='softmax')(branches)

        self.model = Model(input_layer, output_tensor)
        print(self.model.summary())

    def fit_model(self, predictors, target, learning_rate=0.001, loss_function='categorical_crossentropy', epochs=500,
                  batch_size=500, verbose=True, callback_list=[], validation_data=None):
        optimizer = SGD(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        history = self.model.fit(x=predictors, y=target, epochs=epochs, batch_size=batch_size,
                                 validation_data=validation_data,
                                 callbacks=callback_list, verbose=verbose)

    def predict_model(self, x_valid, y_valid, threshold=0.5):
        predicted = self.model.predict(x_valid)
        predicted = predicted[:, 1]
        print(y_valid[:, 1])
        print(predicted)
        predicted = (predicted > threshold).astype(int)
        print('PRECISION ', metrics.precision_score(y_valid[:, 1], predicted))
        print('RECALL ', metrics.recall_score(y_valid[:, 1], predicted))
        print('F1 SCORE ', metrics.f1_score(y_valid[:, 1], predicted))


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
    df['p_churn'] = df['p_churn'].map(float)
    y = to_categorical(df.p_churn)
    del df['p_churn']
    del df['p_id']
    for i in df.columns.values.tolist():
        df[i] = df[i].map(float)
        df[i] = (df[i] - df[i].mean()) / df[i].std()
    x = df.values
    cols = x.shape[1]

    valid = pd.DataFrame(valid[1], columns=columns)
    valid['p_churn'] = valid['p_churn'].map(float)
    y_valid = to_categorical(valid.p_churn)
    del valid['p_churn']
    del valid['p_id']
    for i in valid.columns.values.tolist():
        valid[i] = valid[i].map(float)
        valid[i] = (valid[i] - valid[i].mean()) / valid[i].std()
    valid = valid.values

    start_time = time.time()
    im = InceptionModel(n_cols=cols, activation='relu', prob_dropout=None)
    early_stopping_monitor = EarlyStopping(patience=2)
    im.fit_model(x, y, learning_rate=0.01, loss_function='categorical_crossentropy', epochs=10,
                 batch_size=5000, verbose=True, callback_list=[early_stopping_monitor], validation_data=[valid, y_valid])

    im.predict_model(x_valid=valid, y_valid=y_valid)