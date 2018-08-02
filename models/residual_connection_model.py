import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras import layers, Input, regularizers
from keras.models import Model
from sklearn import metrics
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import pickle

class ResidualConnection(object):
    def __init__(self, n_cols, number_layers=6, node_size=100,
                 prob_dropout=0.1, sparsity_const=0.0002, activation='relu', different_size=None):
        """
        Este modelo ayuda a no perder información reisertandola en futuros layers. Permite resolver dos problemas:

        -Bottlenecks representations: Cuando se usa un modelo Sequential con muchos layers, cada layer es la representación
        construida del anterior. Si un layer es muy pequeño (features con dimensiones bajas), entonces el modelo está 
        restringido a cuanta información puede obtener de este usando una activación. Pensar que si un feature tiene
        valores muy bajos, una vez que se pase la activación es muy dificil que se pueda recuperar hacia valores más altos.
        Esto es, la perdida de información es permanente. Por ello sirve reintroducirlos.

        -Vanishing gradient: Recordar que este comienza desde la función de perdida hacia atrás. Lo que hace es 
        recibir señales de los parámetros que debe modificar. Si la red es muy profunda, esta señal se pierde, haciendo
        el modelo inentrenable. Esto sucede con redes profundas o con muchas secuencias. La conexión residual permite 
        introducir información lineal de manera paralela hacia el main layer stack, ayudando a propagar el gradiente.

        En general es benéfico agregar conexiones residuales a modelos que tienen más de 10 layers.
        Se reconecta un early layer a un later layes (asumiendo que ambas activations tienen el mismo tamaño:
        sino se puede reshapear el early activation de manera lineal, usando un Dense layer sin activación)
        
        :param n_cols: Number of columns of the dataset
        :param number_layers: Number of total layers in the network (without considering the output node)
        :param node_size: Number of nodes per layer
        :param prob_dropout: proportion to dropout
        :param sparsity_const: Restrict some nodes and not all (as PCA), using regularization strategy
        :param activation: Activation function
        :param different_size: Different sizes in the nodes between root and auxiliars
            
        """
        self.n_cols = n_cols
        self.activation = activation
        self.prob_dropout = prob_dropout
        self.number_layers = number_layers
        self.node_size = node_size
        self.sparsity_const = sparsity_const

        input_layer = Input(shape=(n_cols,))
        nodes_range = range(n_cols - node_size*2, node_size - 1, -node_size)
        print(nodes_range)
        # RESIDUAL LAYER
        if sparsity_const is not None:
            residual = layers.Dense(node_size, activation=self.activation, name='residual_layer_' + str(node_size),
                                    activity_regularizer=
                                    regularizers.l1_l2(l1=sparsity_const, l2=sparsity_const))(input_layer)
        else:
            residual = layers.Dense(node_size, activation=self.activation, name='root_layer_' + str(node_size))(input_layer)

        y = residual
        print('residual', y)

            # ROOT LAYERS
        if different_size is None:
            for nodes in nodes_range:
                print(nodes)
                if sparsity_const is not None:
                    print('paso')

                    y = layers.Dense(node_size, activation=self.activation, activity_regularizer=
                    regularizers.l1_l2(l1=sparsity_const, l2=sparsity_const))(y)
                else:
                    y = layers.Dense(node_size, activation=self.activation)(y)
                if self.prob_dropout is not None:
                    y = layers.Dropout(self.prob_dropout)(y)
                print(y)
        else:
            for nodes in nodes_range:
                if sparsity_const is not None:
                    y = residual
                    y = layers.Dense(nodes, activation=self.activation, name='root_layer_'+str(nodes), activity_regularizer=
                    regularizers.l1_l2(l1=sparsity_const, l2=sparsity_const))(y)
                else:
                    y = layers.Dense(nodes + different_size, activation=self.activation, name='root_layer_' + str(nodes))(y)
                if self.prob_dropout is not None:
                    y = layers.Dropout(self.prob_dropout)(y)

            residual = layers.Dense(node_size + different_size)(residual)

        y = layers.add([y, residual])
        output_tensor = layers.Dense(2, activation='softmax')(y)

        self.model = Model(input_layer, output_tensor)
        print(self.model.summary())

    def fit_model(self, predictors, target, learning_rate=0.001, loss_function='categorical_crossentropy', epochs=500,
            batch_size=500, verbose=True, callback_list=[], validation_data=None):
        optimizer = SGD(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        history = self.model.fit(x=predictors, y=target, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
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
    rc = ResidualConnection(n_cols=cols, activation='relu', prob_dropout=None)
    early_stopping_monitor = EarlyStopping(patience=2)
    rc.fit_model(x, y, learning_rate=0.001, loss_function='categorical_crossentropy', epochs=500,
            batch_size=500, verbose=True, callback_list=[early_stopping_monitor], validation_data=[valid, y_valid])

    rc.predict_model(x_valid=valid, y_valid=y_valid)