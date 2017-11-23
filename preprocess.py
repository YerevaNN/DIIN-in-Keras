from datasample import Sample
from util import get_snli_file_path
from util import get_word2vec_file_path
from gensim.models import KeyedVectors

import numpy as np

import sys
import json
import nltk
try:    import cPickle as pickle  # Python 2
except: import _pickle as pickle  # Python 3


class WordVectorGenerator:
    def __init__(self, model_path):
        self.model = KeyedVectors.load_word2vec_format(model_path)

    def __getitem__(self, item):
        try:
            return self.model[item]
        except KeyError:
            return np.random.uniform(low=-1, high=1, size=self.model.vector_size)


class BaseParser(object):
    def __init__(self):
        pass

    def load_data(self, filename):
        raise NotImplementedError("Please implement data loading")

    def parse_one(self, sample):
        raise NotImplementedError("Please implement parsing one sample")

    def parse(self, data):
        samples = []
        for data_sample in data:
            samples.append(self.parse_one(data_sample))
        return samples

    @staticmethod
    def save_data(filename, samples):
        inputs = []
        labels = []
        for sample in samples:
            net_inputs, net_label = sample.to_data_sample()
            inputs.append(net_inputs)
            labels.append(net_label)

        labels = np.array(labels)
        data = (inputs, labels)

        # Serialize large object to file
        n_bytes = 2**31
        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(data)
        print('Serializing object of size:', sys.getsizeof(bytes_out), flush=True)
        with open(filename, 'wb') as f:
            for idx in range(0, n_bytes, max_bytes):
                f.write(bytes_out[idx:idx + max_bytes])

    def load_parse_and_save(self, input_filename, output_filename):
        print('Reading data from:', input_filename, '\tAnd saving the processed result to:', output_filename)
        data = self.load_data(filename=input_filename)
        samples = self.parse(data=data)
        self.save_data(filename=output_filename, samples=samples)


class SNLIParser(BaseParser):
    def __init__(self, word_model):
        super(SNLIParser, self).__init__()
        self.model = word_model

    def load_data(self, filename):
        path = get_snli_file_path()
        with open(path + filename) as f:
            lines = f.readlines()
            text = '[' + ','.join(lines) + ']'
            snli_data = json.loads(text)
            return snli_data

    def parse_one(self, sample):
        # Get data from snli sample, premise=sentence1, hypothesis=sentence2, label=gold_label
        premise = sample['sentence1']
        hypothesis = sample['sentence2']
        label = sample['gold_label']

        # Tokenize words (each word becomes separate element of an array)
        premise_word_vector = nltk.word_tokenize(premise)
        hypothesis_word_vector = nltk.word_tokenize(hypothesis)

        # Convert every word to corresponding word-vector
        premise_word_vector = np.array([self.model[word] for word in premise_word_vector])
        hypothesis_word_vector = np.array([self.model[word] for word in hypothesis_word_vector])

        return Sample(premise=premise,
                      premise_word_vector=premise_word_vector,
                      hypothesis=hypothesis,
                      hypothesis_word_vector=hypothesis_word_vector,
                      label=label)


if __name__ == '__main__':
    print('Loading word2VecModel...', end='', flush=True)
    word2vec_path = get_word2vec_file_path()
    word_vector_model = WordVectorGenerator(word2vec_path)

    snli_parser = SNLIParser(word_vector_model)
    snli_parser.load_parse_and_save(input_filename='snli_1.0_train.jsonl', output_filename='data/train.pkl')
    snli_parser.load_parse_and_save(input_filename='snli_1.0_test.jsonl',  output_filename='data/test.pkl')
    snli_parser.load_parse_and_save(input_filename='snli_1.0_dev.jsonl',   output_filename='data/dev.pkl')
