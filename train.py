import random
import os
import shutil
import numpy as np
from tqdm import tqdm

from model import construct_model
from optimizers.adadelta import AdadeltaL2
from optimizers.sgd import SGDL2
from util import load_train_data

from keras.callbacks import TensorBoard


def train(model, epochs,
          train_data,
          valid_data,
          initial_optimizer=AdadeltaL2(lr=0.1, rho=0.95, epsilon=1e-8),
          secondary_optimizer=SGDL2(lr=3e-4),
          models_save_dir='./models/',
          optimizer_switch_step=30000,
          batch_size=70,
          tensorboard=TensorBoard(),
          shuffle=True):

    model.compile(optimizer=initial_optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

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

        for batch in tqdm(range(0, len(train_data[0]), batch_size)):
            [loss, accuracy] = model.train_on_batch([train_input[batch: batch+batch_size] for train_input in train_inputs],
                                                    train_labels[batch: batch+batch_size])
            logs = {'acc': accuracy, 'loss': loss}
            tensorboard.on_epoch_end(epoch=step, logs=logs)

            step += 1
            no_progress_steps += 1
            if loss < best_loss:
                best_loss = loss
                no_progress_steps = 0

            if no_progress_steps >= optimizer_switch_step:
                optimizer_switch_step = 10000000000           # Never update again
                # params = model.save_weights()
                model.compile(optimizer=secondary_optimizer,  # Compile the model again to use a new optimizer
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

        # Log results to tensorboard and save the model
        [val_loss, val_acc] = model.evaluate(valid_data[:-1], valid_data[-1])
        logs = {'val_loss': val_loss, 'val_acc': val_acc}
        tensorboard.on_epoch_end(epoch=step, logs=logs)
        model.save(filepath=models_save_dir + 'epoch={}-vloss={}-vacc={}.model'.format(epoch, val_loss, val_acc))

    tensorboard.on_train_end('Good Bye!')


if __name__ == '__main__':

    word_embedding_weights = np.load('data/word-vectors.npy')
    train_data = load_train_data('data/dev')
    valid_data = load_train_data('data/test')

    adadelta = AdadeltaL2(lr=0.1, rho=0.95, epsilon=1e-8)
    sgd = SGDL2(lr=3e-4)
    model = construct_model(p=32,
                            h=32,
                            word_embedding_weights=word_embedding_weights,
                            char_pad_size=14,
                            syntactical_feature_size=48)

    tensorboard_dir = './logs'
    # Clean-up if necessary
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir, ignore_errors=True)

    board = TensorBoard(log_dir=tensorboard_dir)
    train(model=model,
          epochs=50,
          train_data=train_data,
          valid_data=valid_data,
          initial_optimizer='adam',
          secondary_optimizer=sgd,
          tensorboard=board)
