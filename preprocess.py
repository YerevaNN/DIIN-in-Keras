from __future__ import print_function

import argparse
import io
import json
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from util import get_snli_file_path, get_word2vec_file_path, ChunkDataManager


def pad(x, maxlen):
    if len(x) <= maxlen:
        pad_width = ((0, maxlen - len(x)), (0, 0))
        return np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    res = x[:maxlen]
    return np.array(res, copy=False)


class BasePreprocessor(object):

    def __init__(self):
        self.word_to_id = {}
        self.char_to_id = {}
        self.vectors = []
        self.part_of_speech_to_id = {}
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

    @staticmethod
    def load_word_vectors(file_path, separator=' ', normalize=True, max_words=None):
        """
        :return: words[], np.array(vectors)
        """
        seen_words = set()
        words = []
        vectors = []
        vector_size = None
        print('Loading', file_path)
        with io.open(file_path, mode='r', encoding='utf-8') as f:
            for line in tqdm(f):
                values = line.replace(' \n', '').split(separator)
                word = values[0]
                if len(values) < 10 or word in seen_words:
                    print('Invalid word:', word)
                    continue

                seen_words.add(word)
                vec = np.asarray(values[1:], dtype='float32')
                if normalize:
                    vec /= np.linalg.norm(vec, ord=2)

                if vector_size is None:
                    vector_size = len(vec)
                elif len(vec) != vector_size:
                    print('Skipping', word)
                    continue

                words.append(word)
                vectors.append(vec)
                if max_words and len(words) >= max_words:
                    break

        vectors = np.array(vectors, dtype='float32', copy=False)
        return words, vectors

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
        all_words = []
        all_parts_of_speech = []
        for file_path in file_paths:
            data = self.load_data(file_path=file_path)

            for sample in tqdm(data):
                premise, hypothesis = self.get_sentences(sample)
                premise_words,    premise_speech    = self.get_words_with_part_of_speech(premise)
                hypothesis_words, hypothesis_speech = self.get_words_with_part_of_speech(hypothesis)
                all_words           += premise_words  + hypothesis_words
                all_parts_of_speech += premise_speech + hypothesis_speech

        self.unique_words           = set(all_words)
        self.unique_parts_of_speech = set(all_parts_of_speech)

    @staticmethod
    def get_not_present_word_vectors(not_present_words, word_vector_size, normalize):
        res_words = []
        res_vectors = []
        for word in not_present_words:
            vec = np.random.uniform(size=word_vector_size)
            if normalize:
                vec /= np.linalg.norm(vec, ord=2)
            res_words.append(word)
            res_vectors.append(vec)
        return res_words, res_vectors

    def init_word_to_vectors(self, vectors_file_path, needed_words, normalize=False, max_loaded_word_vectors=None):
        """
        Initialize:
            {word -> vec} mapping
            {word -> id}  mapping
            [vectors] array
        :param max_loaded_word_vectors: maximum number of words to load from word-vec file
        :param vectors_file_path: file where word-vectors are stored (Glove .txt file)
        :param needed_words: words for which to keep word-vectors
        :param normalize: normalize word vectors
        """
        needed_words = set(needed_words)
        words, self.vectors = self.load_word_vectors(file_path=vectors_file_path,
                                                     normalize=normalize,
                                                     max_words=max_loaded_word_vectors)
        word_vector_size = self.vectors.shape[-1]
        self.vectors = list(self.vectors)

        present_words = needed_words.intersection(words)
        not_present_words = needed_words - present_words
        print('#Present words:', len(present_words), '\t#Not present words', len(not_present_words))

        not_present_words, not_present_vectors = self.get_not_present_word_vectors(not_present_words=not_present_words,
                                                                                   word_vector_size=word_vector_size,
                                                                                   normalize=normalize)
        words, self.vectors = zip(*[(word, vec) for word, vec in zip(words, self.vectors) if word in needed_words])
        words = list(words) + not_present_words
        self.vectors = list(self.vectors) + not_present_vectors

        print('Initializing word mappings...')
        self.word_to_id  = {word: i   for i, word   in enumerate(words)}
        self.vectors = np.array(self.vectors, copy=False)

        assert len(self.word_to_id) == len(self.vectors)
        print(len(self.word_to_id), 'words in total are now initialized!')

    def init_chars(self, words):
        """
        Init char -> id mapping
        """
        chars = set()
        for word in words:
            chars = chars.union(set(word))

        self.char_to_id = {char: i+1 for i, char in enumerate(chars)}
        print('Chars:', chars)

    def init_parts_of_speech(self, parts_of_speech):
        self.part_of_speech_to_id = {part: i+1 for i, part in enumerate(parts_of_speech)}
        print('Parts of speech:', parts_of_speech)

    def save_word_vectors(self, file_path):
        np.save(file_path, self.vectors)

    def get_label(self, sample):
        return NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def label_to_one_hot(self, label):
        label_set = self.get_labels()
        res = np.zeros(shape=(len(label_set)), dtype=np.bool)
        i = label_set.index(label)
        res[i] = 1
        return res

    def parse_sentence(self, sentence, max_words, chars_per_word):
        # Words
        words, parts_of_speech = self.get_words_with_part_of_speech(sentence)
        word_ids = [self.word_to_id[word] for word in words]

        # Syntactical features
        syntactical_features = [self.part_of_speech_to_id[part] for part in parts_of_speech]
        syntactical_one_hot = np.eye(len(self.part_of_speech_to_id) + 2)[syntactical_features]  # Convert to 1-hot

        # Chars
        chars = [[self.char_to_id[c] for c in word] for word in words]
        chars = pad_sequences(chars, maxlen=chars_per_word, padding='post', truncating='post')
        
        return (words, parts_of_speech, np.array(word_ids, copy=False),
                syntactical_features, pad(syntactical_one_hot, max_words),
                pad(chars, max_words))

    def parse_one(self, premise, hypothesis, max_words_p, max_words_h, chars_per_word):
        """
        :param premise: sentence
        :param hypothesis: sentence
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word
        :return: (premise_word_ids, hypothesis_word_ids,
                  premise_chars, hypothesis_chars,
                  premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                  premise_exact_match, hypothesis_exact_match)
        """
        (premise_words, premise_parts_of_speech, premise_word_ids,
         premise_syntactical_features, premise_syntactical_one_hot,
         premise_chars) = self.parse_sentence(sentence=premise, max_words=max_words_p, chars_per_word=chars_per_word)

        (hypothesis_words, hypothesis_parts_of_speech, hypothesis_word_ids,
         hypothesis_syntactical_features, hypothesis_syntactical_one_hot,
         hypothesis_chars) = self.parse_sentence(sentence=hypothesis, max_words=max_words_h, chars_per_word=chars_per_word)

        def calculate_exact_match(source_words, target_words):
            source_words = [word.lower() for word in source_words]
            target_words = [word.lower() for word in target_words]
            target_words = set(target_words)

            res = [(word in target_words) for word in source_words]
            return np.array(res, copy=False)

        premise_exact_match    = calculate_exact_match(premise_words, hypothesis_words)
        hypothesis_exact_match = calculate_exact_match(hypothesis_words, premise_words)

        return (premise_word_ids, hypothesis_word_ids,
                premise_chars, hypothesis_chars,
                premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                premise_exact_match, hypothesis_exact_match)

    def parse(self, input_file_path, max_words_p=33, max_words_h=20, chars_per_word=13):
        """
        :param input_file_path: file to parse data from
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word (padding is applied if not enough)
        :return: (premise_word_ids, hypothesis_word_ids,
                  premise_chars, hypothesis_chars,
                  premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                  premise_exact_match, hypothesis_exact_match)
        """
        # res = [premise_word_ids, hypothesis_word_ids, premise_chars, hypothesis_chars,
        # premise_syntactical_one_hot, hypothesis_syntactical_one_hot, premise_exact_match, hypothesis_exact_match]
        res = [[], [], [], [], [], [], [], [], []]

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

            sample_result = list(sample_inputs) + [label]
            for res_item, parsed_item in zip(res, sample_result):
                res_item.append(parsed_item)

        res[0] = pad_sequences(res[0], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # input_word_p
        res[1] = pad_sequences(res[1], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # input_word_h
        res[6] = pad_sequences(res[6], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # exact_match_p
        res[7] = pad_sequences(res[7], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # exact_match_h
        res = [np.array(item, copy=False) for item in res]
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


def preprocess(p, h, chars_per_word, preprocessor, save_dir, data_paths,
               word_vector_save_path, normalize_word_vectors, max_loaded_word_vectors=None,
               include_word_vectors=True, include_chars=True,
               include_syntactical_features=True, include_exact_match=True):

    preprocessor.get_all_words_with_parts_of_speech([data_path[1] for data_path in data_paths])
    print('Found', len(preprocessor.unique_words), 'unique words')
    print('Found', len(preprocessor.unique_parts_of_speech), 'unique parts of speech')

    # Init mappings of the preprocessor
    preprocessor.init_word_to_vectors(vectors_file_path=get_word2vec_file_path(),
                                      needed_words=preprocessor.unique_words,
                                      normalize=normalize_word_vectors,
                                      max_loaded_word_vectors=max_loaded_word_vectors)
    preprocessor.init_chars(words=preprocessor.unique_words)
    preprocessor.init_parts_of_speech(parts_of_speech=preprocessor.unique_parts_of_speech)

    # Process and save the data
    preprocessor.save_word_vectors(word_vector_save_path)
    for dataset, input_path in data_paths:
        data = preprocessor.parse(input_file_path=input_path,
                                  max_words_p=p,
                                  max_words_h=h,
                                  chars_per_word=chars_per_word)

        # Determine which part of data we need to dump
        if not include_exact_match:             del data[6:8]  # Exact match feature
        if not include_syntactical_features:    del data[4:6]  # Syntactical POS tags
        if not include_chars:                   del data[2:4]  # Character features
        if not include_word_vectors:            del data[0:2]  # Word vectors

        data_saver = ChunkDataManager(save_data_path=os.path.join(save_dir, dataset))
        data_saver.save(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p',              default=32,         help='Maximum words in premise',            type=int)
    parser.add_argument('--h',              default=32,         help='Maximum words in hypothesis',         type=int)
    parser.add_argument('--chars_per_word', default=16,         help='Number of characters in one word',    type=int)
    parser.add_argument('--max_word_vecs',  default=None,       help='Maximum number of word vectors',      type=int)
    parser.add_argument('--save_dir',       default='data/',    help='Save directory of data',              type=str)
    parser.add_argument('--dataset',        default='snli',     help='Which preprocessor to use',           type=str)
    parser.add_argument('--word_vec_path',  default='data/word-vectors.npy', help='Save path word vectors', type=str)
    parser.add_argument('--normalize_word_vectors',      action='store_true')
    parser.add_argument('--omit_word_vectors',           action='store_true')
    parser.add_argument('--omit_chars',                  action='store_true')
    parser.add_argument('--omit_syntactical_features',   action='store_true')
    parser.add_argument('--omit_exact_match',            action='store_true')
    args = parser.parse_args()

    if args.dataset == 'snli':
        snli_preprocessor = SNLIPreprocessor()
        path = get_snli_file_path()
        train_path = os.path.join(path, 'snli_1.0_train.jsonl')
        test_path  = os.path.join(path, 'snli_1.0_test.jsonl')
        dev_path   = os.path.join(path, 'snli_1.0_dev.jsonl')

        preprocess(p=args.p, h=args.h, chars_per_word=args.chars_per_word,
                   preprocessor=snli_preprocessor,
                   save_dir=args.save_dir,
                   data_paths=[('train', train_path), ('test', test_path), ('dev', dev_path)],
                   normalize_word_vectors=args.normalize_word_vectors,
                   word_vector_save_path=args.word_vec_path,
                   max_loaded_word_vectors=args.max_word_vecs,
                   include_word_vectors=not args.omit_word_vectors,
                   include_chars=not args.omit_chars,
                   include_syntactical_features=not args.omit_syntactical_features,
                   include_exact_match=not args.omit_exact_match)
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')
