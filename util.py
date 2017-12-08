from __future__ import print_function

import os

from keras import backend as K
from os import path

from gensim.scripts.glove2word2vec import glove2word2vec
from keras.utils import get_file


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


def broadcast_last_axis(x):
    """
    Accepts tensor of shape (batch, a, b, ..., k)
     :returns broadcasted tensor of shape (batch, a, b, ..., k, a)
    """
    z = K.identity(x) * 0
    z = K.expand_dims(z)
    s = z[(Ellipsis,) + (0,) * (K.ndim(z) - 1)]
    s = K.expand_dims(s, axis=0)
    res = K.dot(z, s)
    res = K.permute_dimensions(res, pattern=(0, 2, 1, 3))
    return res + x
