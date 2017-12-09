import numpy as np

from model import construct_model
from optimizers.adadelta import AdadeltaL2
from optimizers.sgd import SGDL2
from util import load_train_data

from keras.callbacks import TensorBoard


def train(model, epochs,
          train_inputs, train_labels,
          valid_inputs, valid_labels,
          initial_optimizer, secondary_optimizer,
          optimizer_switch_step=30000,
          batch_size=70):

    model.fit(x=train_inputs,
              y=train_labels,
              epochs=epochs,
              validation_data=(valid_inputs, valid_labels),
              callbacks=[TensorBoard()])


if __name__ == '__main__':

    word_embedding_weights = np.load('data/word-vectors.npy')
    train_data = load_train_data('data/train')
    valid_data = load_train_data('data/test')

    adadelta = AdadeltaL2(lr=0.1, rho=0.95, epsilon=1e-8)
    sgd = SGDL2(lr=3e-4)

    model = construct_model(p=32,
                            h=32,
                            word_embedding_weights=word_embedding_weights,
                            char_pad_size=14,
                            syntactical_feature_size=48)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    train(model=model,
          epochs=50,
          train_inputs=train_data[:-1],
          train_labels=train_data[-1],
          valid_inputs=valid_data[:-1],
          valid_labels=valid_data[-1],
          initial_optimizer=adadelta,
          secondary_optimizer=sgd)
