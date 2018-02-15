from __future__ import print_function

import argparse
import os
import random

import numpy as np
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, Adagrad
from tqdm import tqdm

from model import DIIN
from optimizers.l2optimizer import L2Optimizer
from util import ChunkDataManager


class Gym(object):
    def __init__(self,
                 model,
                 train_data,
                 test_data,
                 dev_data,
                 optimizers,
                 logger,
                 models_save_dir):

        self.model = model
        self.logger = logger

        ''' Data '''
        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = dev_data
        self.model_save_dir = models_save_dir
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        ''' Optimizers '''
        self.optimizers = optimizers
        self.optimizer_id = -1
        self.current_optimizer = None
        self.current_switch_step = -1

    def switch_optimizer(self):
        self.optimizer_id += 1
        if self.optimizer_id >= len(self.optimizers):
            print('Finished training...')
            exit(0)

        self.current_optimizer, self.current_switch_step = self.optimizers[self.optimizer_id]
        self.model.compile(optimizer=self.current_optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.logger.set_model(self.model)
        print('Switching to number {} optimizer'.format(self.current_optimizer))

    def train(self, batch_size=70, eval_interval=500, shuffle=True):
        print('train:\t', [d.shape for d in self.train_data])
        print('test:\t',  [d.shape for d in self.test_data])
        print('dev:\t',   [d.shape for d in self.dev_data])

        # Initialize optimizer
        self.switch_optimizer()
        self.model.summary()

        # Start training
        train_step, eval_step, no_progress_steps = 0, 0, 0
        train_batch_start = 0
        best_loss = 1000.

        while True:
            if shuffle:
                random.shuffle(list(zip(train_data)))
            train_inputs = train_data[:-1]
            train_labels = train_data[-1]

            # Evaluate
            test_loss, dev_loss = self.evaluate(eval_step=eval_step, batch_size=batch_size)
            eval_step += 1

            # Switch optimizer if it's necessary
            no_progress_steps += 1
            if dev_loss < best_loss:
                best_loss = dev_loss
                no_progress_steps = 0

            if no_progress_steps >= self.current_switch_step:
                self.switch_optimizer()
                no_progress_steps = 0

            # Train eval_interval times
            for _ in tqdm(range(eval_interval)):
                [loss, acc] = model.train_on_batch(
                    [train_input[train_batch_start: train_batch_start + batch_size] for train_input in train_inputs],
                    train_labels[train_batch_start: train_batch_start + batch_size])
                self.logger.on_epoch_end(epoch=train_step, logs={'train_acc': acc, 'train_loss': loss})
                train_step += 1
                train_batch_start += batch_size
                if train_batch_start > len(train_inputs[0]):
                    train_batch_start = 0
                    # Shuffle the data after the epoch ends
                    if shuffle:
                        random.shuffle(list(zip(train_data)))

    def evaluate(self, eval_step, batch_size=None):
        [test_loss, test_acc] = model.evaluate(self.test_data[:-1], self.test_data[-1], batch_size=batch_size)
        [dev_loss,  dev_acc]  = model.evaluate(self.dev_data[:-1],  self.dev_data[-1],  batch_size=batch_size)
        self.logger.on_epoch_end(epoch=eval_step, logs={'test_acc': test_acc, 'test_loss': test_loss})
        self.logger.on_epoch_end(epoch=eval_step, logs={'dev_acc':  dev_acc,  'dev_loss':  dev_loss})
        model.save(self.model_save_dir + 'epoch={}-tloss={}-tacc={}.model'.format(eval_step, test_loss, test_acc))

        return test_loss, dev_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',                  default=70,        help='Batch size',                              type=int)
    parser.add_argument('--eval_interval',               default=500,       help='Evaluation Interval (#batches)',          type=int)
    parser.add_argument('--char_embed_size',             default=8,         help='Size of character embedding',             type=int)
    parser.add_argument('--char_conv_filters',           default=100,       help='Number of character conv filters',        type=int)
    parser.add_argument('--char_conv_kernel',            default=5,         help='Size of char convolution kernel',         type=int)
    parser.add_argument('--dropout_initial_keep_rate',   default=1.,        help='Initial keep rate of decaying dropout',   type=float)
    parser.add_argument('--dropout_decay_rate',          default=0.977,     help='Decay rate of dropout',                   type=float)
    parser.add_argument('--dropout_decay_interval',      default=10000,     help='Dropout decay interval',                  type=int)
    parser.add_argument('--l2_full_step',                default=100000,    help='Number of steps for full L2 penalty',     type=float)
    parser.add_argument('--l2_full_ratio',               default=9e-5,      help='L2 full penalty',                         type=float)
    parser.add_argument('--l2_diference_penalty',        default=1e-3,      help='L2 penalty applied on weight difference', type=float)
    parser.add_argument('--first_scale_down_ratio',      default=0.3,       help='First scale down ratio (DenseNet)',       type=float)
    parser.add_argument('--transition_scale_down_ratio', default=0.5,       help='Transition scale down ratio (DenseNet)',  type=float)
    parser.add_argument('--growth_rate',                 default=20,        help='Growth rate (DenseNet)',                  type=int)
    parser.add_argument('--layers_per_dense_block',      default=8,         help='Layers in one Dense block (DenseNet)',    type=int)
    parser.add_argument('--dense_blocks',                default=3,         help='Number of Dense blocks (DenseNet)',       type=int)
    parser.add_argument('--labels',                      default=3,         help='Number of output labels',                 type=int)
    parser.add_argument('--load_dir',                    default='data',    help='Directory of the data',   type=str)
    parser.add_argument('--models_dir',                  default='models/', help='Where to save models',    type=str)
    parser.add_argument('--logdir',                      default='logs',    help='Tensorboard logs dir',    type=str)
    parser.add_argument('--word_vec_path', default='data/word-vectors.npy', help='Save path word vectors',  type=str)
    parser.add_argument('--omit_word_vectors',           action='store_true')
    parser.add_argument('--omit_chars',                  action='store_true')
    parser.add_argument('--omit_syntactical_features',   action='store_true')
    parser.add_argument('--omit_exact_match',            action='store_true')
    parser.add_argument('--train_word_embeddings',       action='store_true')
    args = parser.parse_args()

    ''' Prepare data '''
    word_embedding_weights = np.load(args.word_vec_path)
    train_data = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'train')).load()
    test_data  = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'test')).load()
    dev_data   = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'dev')).load()

    ''' Getting dimensions of the input '''
    chars_per_word = train_data[3].shape[-1] if not args.omit_chars else 0
    syntactical_feature_size = train_data[5].shape[-1] if not args.omit_syntactical_features else 0

    ''' Prepare the model and optimizers '''
    adam = L2Optimizer(Adam(), args.l2_full_step, args.l2_full_ratio, args.l2_diference_penalty)
    adagrad = L2Optimizer(Adagrad(), args.l2_full_step, args.l2_full_ratio, args.l2_diference_penalty)
    sgd = L2Optimizer(SGD(lr=3e-3), args.l2_full_step, args.l2_full_ratio, args.l2_diference_penalty)

    model = DIIN(p=train_data[0].shape[-1],  # or None
                 h=train_data[1].shape[-1],  # or None
                 include_word_vectors=not args.omit_word_vectors,
                 word_embedding_weights=word_embedding_weights,
                 train_word_embeddings=args.train_word_embeddings,
                 include_chars=not args.omit_chars,
                 chars_per_word=chars_per_word,
                 char_embedding_size=args.char_embed_size,
                 char_conv_filters=args.char_conv_filters,
                 char_conv_kernel_size=args.char_conv_kernel,
                 include_syntactical_features=not args.omit_syntactical_features,
                 syntactical_feature_size=syntactical_feature_size,
                 include_exact_match=not args.omit_exact_match,
                 dropout_initial_keep_rate=args.dropout_initial_keep_rate,
                 dropout_decay_rate=args.dropout_decay_rate,
                 dropout_decay_interval=args.dropout_decay_interval,
                 first_scale_down_ratio=args.first_scale_down_ratio,
                 transition_scale_down_ratio=args.transition_scale_down_ratio,
                 growth_rate=args.growth_rate,
                 layers_per_dense_block=args.layers_per_dense_block,
                 nb_dense_blocks=args.dense_blocks,
                 nb_labels=args.labels)

    ''' Initialize Gym for training '''
    gym = Gym(model=model,
              train_data=train_data, test_data=test_data, dev_data=dev_data,
              optimizers=[(adam, 3), (adagrad, 4), (sgd, 15)],
              logger=TensorBoard(log_dir=args.logdir),
              models_save_dir=args.models_dir)

    gym.train(batch_size=args.batch_size, eval_interval=args.eval_interval, shuffle=True)
