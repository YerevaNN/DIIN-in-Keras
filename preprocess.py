from __future__ import print_function

import argparse
import io
import json
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from util import get_snli_file_path, ChunkDataManager
from util import get_word2vec_file_path


def pad(x, maxlen):
    if len(x) <= maxlen:
        pad_width = ((0, maxlen - len(x)), (0, 0))
        return np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    res = x[:maxlen]
    return np.array(res)


class BasePreprocessor(object):

    def __init__(self):
        self.word_to_vec = {}
        self.word_to_id = {}
        self.char_to_id = {}
        self.vectors = []
        self.part_of_speech_to_id = {}
        self.all_words = []
        self.all_parts_of_speech = []
        self.unique_words = set()
        self.unique_parts_of_speech = set()

    @staticmethod
    def load_data(file_path):
        """
        Load jsonl file by default
        """
        with open(file_path) as f:
            lines = f.readlines()
            text = '[' + ','.join(lines) + ']'
            return json.loads(text)

    def get_words_with_part_of_speech(self, sentence):
        """
        :return: words, parts_of_speech
        """
        raise NotImplementedError

    def get_sentences(self, sample):
        """
        :param sample: sample from data
        :return: premise, hypothesis
        """
        raise NotImplementedError

    def get_all_words_with_parts_of_speech(self, file_paths):
        """
        :param file_paths: paths to files where the data is stored
        :return: words, parts_of_speech
        """
        for file_path in file_paths:
            data = self.load_data(file_path=file_path)

            for sample in tqdm(data):
                premise, hypothesis = self.get_sentences(sample)
                premise_words,    premise_speech    = self.get_words_with_part_of_speech(premise)
                hypothesis_words, hypothesis_speech = self.get_words_with_part_of_speech(hypothesis)
                self.all_words           += premise_words  + hypothesis_words
                self.all_parts_of_speech += premise_speech + hypothesis_speech

        self.unique_words           = set(self.all_words)
        self.unique_parts_of_speech = set(self.all_parts_of_speech)

    def init_word_to_vecs(self, vectors_file_path, needed_words, normalize=False):
        """
        Initialize:
            {word -> vec} mapping
            {word -> id}  mapping
            [vectors] array
        :param vectors_file_path: file where word-vectors are stored (Glove .txt file)
        :param needed_words: words for which to keep word-vectors
        :param normalize: normalize word vectors
        """
        needed_words = set(needed_words)
        self.word_to_vec = {}
        self.word_to_id = {}
        self.vectors = []

        with io.open(vectors_file_path, mode='r', encoding='utf-8') as f:
            for line in tqdm(f):
                values = line.split(' ')
                word = values[0]

                if word in needed_words:
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.word_to_vec[word] = coefs

        word_vector_size = len(self.word_to_vec['hello'])
        present_words = set(self.word_to_vec.keys())
        not_present_words = needed_words - present_words
        print('#Present words:', len(present_words), '\t#Not present words', len(not_present_words))
        print('Inserting not present words to the list as uniform-random vectors')

        for word in not_present_words:
            self.word_to_vec[word] = np.random.uniform(size=word_vector_size)

        if normalize:
            for word, coefs in self.word_to_vec.items():
                self.word_to_vec[word] = coefs / np.linalg.norm(coefs, ord=2)

        for i, (word, vec) in enumerate(self.word_to_vec.items()):
            self.word_to_id[word] = i
            self.vectors.append(vec)

        self.vectors = np.array(self.vectors)

    def init_chars(self, words):
        """
        Init char -> id mapping
        """
        chars = set()
        self.char_to_id = {}
        for word in words:
            chars = chars.union(set(word))

        for i, c in enumerate(chars):
            self.char_to_id[c] = i + 1
            print(c, end=' ')
        print('\n')

    def init_parts_of_speech(self, parts_of_speech):
        """
        :param parts_of_speech:
        :return:
        """
        self.part_of_speech_to_id = {}
        for i, part in enumerate(parts_of_speech):
            self.part_of_speech_to_id[part] = i + 1
            print(part, end=' ')
        print('\n')

    def init_mappings(self):
        snli_preprocessor.init_word_to_vecs(vectors_file_path=get_word2vec_file_path(), needed_words=self.unique_words)
        snli_preprocessor.init_chars(words=self.unique_words)
        snli_preprocessor.init_parts_of_speech(parts_of_speech=self.unique_parts_of_speech)

    def save_word_vectors(self, file_path):
        np.save(file_path, self.vectors)

    def get_label(self, sample):
        return NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def label_to_one_hot(self, label):
        label_set = self.get_labels()
        res = np.zeros(shape=(len(label_set)), dtype=np.bool)
        for i, l in enumerate(label_set):
            if label == l:
                res[i] = 1
                return res
        raise ValueError('Illegal label provided')

    def parse_one(self, premise, hypothesis, max_words_p, max_words_h, chars_per_word):
        """
        :param premise: sentence
        :param hypothesis: sentence
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word
        :return: (premise_word_ids,    premise_chars,    syntactical_premise,
                  hypothesis_word_ids, hypothesis_chars, syntactical_hypothesis)
        """
        # Words
        premise_words,      premise_parts_of_speech    = self.get_words_with_part_of_speech(premise)
        hypothesis_words,   hypothesis_parts_of_speech = self.get_words_with_part_of_speech(hypothesis)
        premise_word_ids    = [self.word_to_id[word] for word in premise_words]
        hypothesis_word_ids = [self.word_to_id[word] for word in hypothesis_words]

        # Syntactical features
        syntactical_premise    = [self.part_of_speech_to_id[part] for part in premise_parts_of_speech]
        syntactical_hypothesis = [self.part_of_speech_to_id[part] for part in hypothesis_parts_of_speech]
        premise_hot    = list(np.eye(len(self.part_of_speech_to_id) + 2)[syntactical_premise])       # Convert to 1-hot
        hypothesis_hot = list(np.eye(len(self.part_of_speech_to_id) + 2)[syntactical_hypothesis])    # Convert to 1-hot

        syntactical_premise = []
        syntactical_hypothesis = []
        premise_lower_words = [word.lower() for word in premise_words]
        hypothesis_lower_words = [word.lower() for word in hypothesis_words]
        for word, hot in zip(premise_lower_words, premise_hot):
            l_hot = list(hot)
            l_hot.append(word in hypothesis_lower_words)
            syntactical_premise.append(np.array(l_hot))
        for word, hot in zip(hypothesis_lower_words, hypothesis_hot):
            l_hot = list(hot)
            l_hot.append(word in premise_lower_words)
            syntactical_hypothesis.append(np.array(l_hot))
        syntactical_premise    = np.array(syntactical_premise)
        syntactical_hypothesis = np.array(syntactical_hypothesis)

        # Chars
        premise_chars = []
        hypothesis_chars = []
        for word in premise_words:     premise_chars.append(   [self.char_to_id[c] for c in word])
        for word in hypothesis_words:  hypothesis_chars.append([self.char_to_id[c] for c in word])
        premise_chars    = pad_sequences(premise_chars,    maxlen=chars_per_word, padding='post', truncating='post')
        hypothesis_chars = pad_sequences(hypothesis_chars, maxlen=chars_per_word, padding='post', truncating='post')

        return (np.array(premise_word_ids),    pad(premise_chars,    max_words_p), pad(syntactical_premise,    max_words_p),
                np.array(hypothesis_word_ids), pad(hypothesis_chars, max_words_h), pad(syntactical_hypothesis, max_words_h))

    def parse(self, input_file_path, data_saver, max_words_p=33, max_words_h=20, chars_per_word=13):
        """
        :param input_file_path: file to parse data from
        :param data_saver: manager for saving results
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word (padding is applied if not enough)
        :return: (premise_word_ids,    premise_chars,    syntactical_premise,
                  hypothesis_word_ids, hypothesis_chars, syntactical_hypothesis)
        """
        # res = [input_word_p, input_char_p, input_syn_p, input_word_h, input_char_h, input_syn_h, labels]
        res = [[], [], [], [], [], [], []]

        data = self.load_data(input_file_path)
        for sample in tqdm(data):
            # As stated in paper: The labels are "entailment", "neutral", "contradiction" and "-".
            # "-"  shows that annotators can't reach consensus with each other, thus removed during training and testing
            label = self.get_label(sample=sample)
            if label == '-':
                continue
            premise, hypothesis = self.get_sentences(sample=sample)
            sample_inputs = self.parse_one(premise, hypothesis,
                                           max_words_h=max_words_h, max_words_p=max_words_p,
                                           chars_per_word=chars_per_word)
            label = self.label_to_one_hot(label=label)

            sample_result = list(sample_inputs)
            sample_result.append(label)
            for res_item, parsed_item in zip(res, sample_result):
                res_item.append(parsed_item)

        res[0] = pad_sequences(res[0], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[3] = pad_sequences(res[3], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h
        res = (np.array(item) for item in res)
        data_saver.save(res)
        return res


class SNLIPreprocessor(BasePreprocessor):
    def get_words_with_part_of_speech(self, sentence):
        parts = sentence.split('(')
        words = []
        parts_of_speech = []
        for p in parts:
            if ')' in p:
                res = p.split(' ')
                parts_of_speech.append(res[0])
                words.append(res[1].replace(')', ''))
        return words, parts_of_speech

    def get_sentences(self, sample):
        return sample['sentence1_parse'], sample['sentence2_parse']

    def get_label(self, sample):
        return sample['gold_label']

    def get_labels(self):
        return 'entailment', 'contradiction', 'neutral'


def preprocess(p, h, chars_per_word, preprocessor, save_dir, data_paths, word_vector_save_path):
    preprocessor.get_all_words_with_parts_of_speech([data_path[1] for data_path in data_paths])
    print('Found', len(preprocessor.unique_words), 'unique words')
    print('Found', len(preprocessor.unique_parts_of_speech), 'unique parts of speech')

    preprocessor.init_mappings()
    preprocessor.save_word_vectors(word_vector_save_path)
    for dataset, input_path in data_paths:
        preprocessor.parse(input_file_path=input_path,
                           data_saver=ChunkDataManager(save_data_path=os.path.join(save_dir, dataset)),
                           max_words_p=p,
                           max_words_h=h,
                           chars_per_word=chars_per_word)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p',              default=32,         help='Maximum words in premise',            type=int)
    parser.add_argument('--h',              default=32,         help='Maximum words in hypothesis',         type=int)
    parser.add_argument('--chars_per_word', default=16,         help='Number of characters in one word',    type=int)
    parser.add_argument('--save_dir',       default='data/',    help='Save directory of data',              type=str)
    parser.add_argument('--dataset',        default='snli',     help='Which preprocessor to use',           type=str)
    parser.add_argument('--word_vec_path',  default='data/word-vectors.npy', help='Save path word vectors', type=str)
    args = parser.parse_args()

    if args.dataset == 'snli':
        snli_preprocessor = SNLIPreprocessor()
        path = get_snli_file_path()
        train_path = path + 'snli_1.0_train.jsonl'
        test_path  = path + 'snli_1.0_test.jsonl'
        dev_path   = path + 'snli_1.0_dev.jsonl'

        preprocess(p=args.p, h=args.h, chars_per_word=args.chars_per_word,
                   preprocessor=snli_preprocessor,
                   save_dir=args.save_dir,
                   data_paths=[('train', train_path), ('test', test_path), ('dev', dev_path)],
                   word_vector_save_path=args.word_vec_path)
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')
