from __future__ import print_function

import argparse
import os
import random
import shutil

import numpy as np
from keras.callbacks import TensorBoard
from tqdm import tqdm

from model import DIIN
from optimizers.adadelta import AdadeltaL2
from optimizers.sgd import SGDL2
from util import ChunkDataManager


def train(model,
          train_data,
          test_data,
          dev_data,
          initial_optimizer,
          secondary_optimizer,
          logger,
          models_save_dir,
          optimizer_switch_step=30000,
          epochs=30,
          batch_size=70,
          shuffle=True):

    print('train:\t', [d.shape for d in train_data])
    print('test:\t',  [d.shape for d in test_data])
    print('dev:\t',   [d.shape for d in dev_data])

    model.compile(optimizer=initial_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    step, no_progress_steps = 0, 0
    best_loss = 1000.
    logger.set_model(model)

    # Start training
    for epoch in range(epochs):
        if shuffle:
            random.shuffle(list(zip(train_data)))
        train_inputs = train_data[:-1]
        train_labels = train_data[-1]

        # Log results to tensorboard and save the model
        [test_loss, test_acc] = model.evaluate(test_data[:-1], test_data[-1], batch_size=batch_size)
        [dev_loss,  dev_acc]  = model.evaluate(dev_data[:-1],  dev_data[-1],  batch_size=batch_size)
        logger.on_epoch_end(epoch=epoch, logs={'test_acc': test_acc, 'test_loss': test_loss})
        logger.on_epoch_end(epoch=epoch, logs={'dev_acc': dev_acc,   'dev_loss': dev_loss})
        model.save(filepath=models_save_dir + 'epoch={}-tloss={}-tacc={}.model'.format(epoch, test_loss, test_acc))

        # Switch optimizer if it's necessary
        no_progress_steps += 1
        if test_loss < best_loss:
            best_loss = test_loss
            no_progress_steps = 0

        if no_progress_steps >= optimizer_switch_step:
            print('Switching to the secondary optimizer...')
            optimizer_switch_step = 10000000000             # Never update again
            model.compile(optimizer=secondary_optimizer,    # Compile the model again to use a new optimizer
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            print('Recompiled the model!')

        # Train one epoch
        for batch in tqdm(range(0, len(train_data[0]), batch_size)):
            [loss, acc] = model.train_on_batch([train_input[batch: batch+batch_size] for train_input in train_inputs],
                                               train_labels[batch: batch+batch_size])
            logger.on_epoch_end(epoch=step, logs={'acc': acc, 'loss': loss})
            step += 1

    logger.on_train_end('Good Bye!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',         default=70,     help='Batch size',                          type=int)
    parser.add_argument('--optimizer_switch',   default=1,      help='Switch step of optimizer',            type=int)
    parser.add_argument('--char_embed_size',    default=8,      help='Size of character embedding',         type=int)
    parser.add_argument('--char_conv_filters',  default=100,    help='Number of character conv filters',    type=int)
    parser.add_argument('--load_dir',           default='data',             help='Directory of the data',   type=str)
    parser.add_argument('--models_dir',         default='models/',          help='Where to save models',    type=str)
    parser.add_argument('--logdir',             default='logs',             help='Tensorboard logs dir',    type=str)
    parser.add_argument('--word_vec_path', default='data/word-vectors.npy', help='Save path word vectors',  type=str)
    args = parser.parse_args()

    ''' Prepare data '''
    word_embedding_weights = np.load(args.word_vec_path)
    train_data = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'train')).load()
    test_data  = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'test')).load()
    dev_data   = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'dev')).load()

    ''' Getting dimensions of the input '''
    chars_per_word = train_data[1].shape[-1]
    syntactical_feature_size = train_data[2].shape[-1]
    assert chars_per_word == train_data[4].shape[-1]
    assert syntactical_feature_size == train_data[5].shape[-1]

    ''' Prepare the model and optimizers '''
    adadelta = AdadeltaL2(lr=0.5, rho=0.95, epsilon=1e-8)
    sgd = SGDL2(lr=3e-4)
    model = DIIN(p=train_data[0].shape[-1],  # or None
                 h=train_data[3].shape[-1],  # or None
                 word_embedding_weights=word_embedding_weights,
                 chars_per_word=chars_per_word,
                 syntactical_feature_size=syntactical_feature_size,
                 char_embedding_size=args.char_embed_size,
                 char_conv_filters=args.char_conv_filters)

    # Prepare directory for models
    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)

    # Clean-up tensorboard dir if necessary
    tensorboard_dir = args.logdir
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir, ignore_errors=True)

    board = TensorBoard(log_dir=tensorboard_dir)
    train(model=model,
          epochs=33,
          train_data=train_data,
          test_data=test_data,
          dev_data=dev_data,
          initial_optimizer=adadelta,
          secondary_optimizer=sgd,
          batch_size=args.batch_size,
          optimizer_switch_step=args.optimizer_switch,
          models_save_dir=args.models_dir,
          logger=board)
