from __future__ import print_function

import os
import random
import shutil

import numpy as np
from keras.callbacks import TensorBoard
from tqdm import tqdm

from metrics import precision, recall
from model import construct_model
from optimizers.adadelta import AdadeltaL2
from optimizers.sgd import SGDL2
from util import load_train_data


def train(model, epochs,
          train_data,
          valid_data,
          initial_optimizer,
          secondary_optimizer,
          models_save_dir='./models/',
          optimizer_switch_step=30000,
          batch_size=70,
          tensorboard=TensorBoard(),
          shuffle=True):

    print('train:')
    [print(d.shape) for d in train_data]
    print('valid:')
    [print(d.shape) for d in valid_data]

    model.compile(optimizer=initial_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', precision, recall])

    step = 0
    best_loss = 1000.
    no_progress_steps = 0
    tensorboard.set_model(model)

    # Start training
    for epoch in range(epochs):
        if shuffle:
            random.shuffle(list(zip(train_data)))
        train_inputs = train_data[:-1]
        train_labels = train_data[-1]

        # Log results to tensorboard and save the model
        [val_loss, val_acc, val_prec, val_rec] = model.evaluate(valid_data[:-1], valid_data[-1])
        tensorboard.on_epoch_end(epoch=epoch, logs={'val_loss': val_loss,
                                                    'val_acc': val_acc,
                                                    'val_precision': val_prec,
                                                    'val_recall': val_rec})
        model.save(filepath=models_save_dir + 'epoch={}-vloss={}-vacc={}.model'.format(epoch, val_loss, val_acc))

        for batch in tqdm(range(0, len(train_data[0]), batch_size)):
            [loss, accuracy, prec, rec] = model.train_on_batch([train_input[batch: batch+batch_size] for train_input in train_inputs],
                                                               train_labels[batch: batch+batch_size])
            tensorboard.on_epoch_end(epoch=step, logs={'acc': accuracy,
                                                       'loss': loss,
                                                       'precision': prec,
                                                       'recall': rec})
            step += 1
            no_progress_steps += 1
            if loss < best_loss:
                best_loss = loss
                no_progress_steps = 0

            if no_progress_steps >= optimizer_switch_step:
                print('Switching to the secondary optimizer...')
                optimizer_switch_step = 10000000000           # Never update again
                # params = model.save_weights()
                model.compile(optimizer=secondary_optimizer,  # Compile the model again to use a new optimizer
                              loss='categorical_crossentropy',
                              metrics=['accuracy', precision, recall])
                print('Recompiled the model!')

    tensorboard.on_train_end('Good Bye!')


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

    # Prepare directory for models
    models_save_dir = './models/'
    if not os.path.exists(models_save_dir):
        os.mkdir(models_save_dir)

    # Clean-up tensorboard dir if necessary
    tensorboard_dir = './logs'
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir, ignore_errors=True)

    board = TensorBoard(log_dir=tensorboard_dir)
    train(model=model,
          epochs=20,
          train_data=train_data,
          valid_data=valid_data,
          initial_optimizer=adadelta,
          secondary_optimizer=sgd,
          models_save_dir=models_save_dir,
          tensorboard=board)
