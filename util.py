from __future__ import print_function

import os
import numpy as np

from keras import backend as K
from os import path

from keras.utils import get_file
from tqdm import tqdm
try:    import cPickle as pickle  # Python 2
except: import _pickle as pickle  # Python 3


def get_snli_file_path():
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')
    download_url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    snli_dir = cache_dir + '/snli_1.0/'

    if os.path.exists(snli_dir):
        return snli_dir

    get_file('/tmp/snli_1.0.zip',
             origin=download_url,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    return snli_dir


def get_word2vec_file_path():
    download_url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')
    glove_file_path = cache_dir + '/glove.840B.300d.txt'

    if path.exists(glove_file_path):
        return glove_file_path

    filename = '/tmp/glove.zip'
    get_file(filename,
             origin=download_url,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    os.remove(filename)
    return glove_file_path


def save_train_data(directory, data):
    if not os.path.exists(directory):
        os.mkdir(directory)
    for i, item in tqdm(enumerate(data)):
        np.save(directory + '/' + str(i) + '.npy', item)


def load_train_data(directory):
    data = []
    for file in tqdm(sorted(os.listdir(directory))):
        if not file.endswith('.npy'):
            continue
        data.append(np.load(directory + '/' + file))
    return data


def broadcast_last_axis(x):
    """
    :param x tensor of shape (batch, a, b)
     :returns broadcasted tensor of shape (batch, a, b, a)
    """
    y = K.expand_dims(x, 1)
    y = K.permute_dimensions(y, (0, 1, 3, 2))
    return (K.expand_dims(x) + y) / 2
