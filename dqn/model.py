from keras.layers import Input
from keras.layers.core import Dense, Flatten
from keras.models import Model

floatX = 'float32'


def build_dense(state_shape, nb_units, nb_actions, nb_channels, remove_features=False):
    if remove_features:
        state_shape = state_shape[: -1] + [state_shape[-1] - nb_channels + 1]
    input_dim = tuple(state_shape)
    states = Input(shape=input_dim, dtype=floatX, name='states')
    flatten = Flatten()(states)
    hid = Dense(output_dim=nb_units, init='he_uniform', activation='relu', name='hidden')(flatten)
    out = Dense(output_dim=nb_actions, init='he_uniform', activation='linear', name='out')(hid)
    return Model(input=states, output=out)
