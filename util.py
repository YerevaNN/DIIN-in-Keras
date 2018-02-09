from __future__ import print_function

import os
from os import path

import numpy as np
from keras import backend as K
from keras.utils import get_file
from tqdm import tqdm


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


def get_word2vec_file_path(file_path):
    if file_path is not None and path.exists(file_path):
        return file_path

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


class ChunkDataManager(object):
    def __init__(self, load_data_path=None, save_data_path=None):
        self.load_data_path = load_data_path
        self.save_data_path = save_data_path

    def load(self):
        data = []
        for file in tqdm(sorted(os.listdir(self.load_data_path))):
            if not file.endswith('.npy'):
                continue
            data.append(np.load(self.load_data_path + '/' + file))
        return data

    def save(self, data):
        if not os.path.exists(self.save_data_path):
            os.mkdir(self.save_data_path)
        print('Saving data of shapes:', [item.shape for item in data])
        for i, item in tqdm(enumerate(data)):
            np.save(self.save_data_path + '/' + str(i) + '.npy', item)


def broadcast_last_axis(x):
    """
    :param x tensor of shape (batch, a, b)
    :returns broadcasted tensor of shape (batch, a, b, a)
    """
    y = K.expand_dims(x, 1) * 0
    y = K.permute_dimensions(y, (0, 1, 3, 2))
    return y + K.expand_dims(x)
