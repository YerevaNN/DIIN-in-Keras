from __future__ import print_function

import os

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
    word2vec_path = cache_dir + '/glove.vec'

    if path.exists(word2vec_path):
        return word2vec_path

    filename = '/tmp/glove.zip'
    get_file(filename,
             origin=download_url,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    print('Converting GloVe (.txt) to word2vec (.vec) format...', end='')
    glove2word2vec(glove_file_path, word2vec_path)
    os.remove(glove_file_path)
    os.remove(filename)
    print('Done')

    return word2vec_path
