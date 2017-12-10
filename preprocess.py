import json
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from util import get_snli_file_path
from util import get_word2vec_file_path


def load_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        text = '[' + ','.join(lines) + ']'
        return json.loads(text)


def get_words_with_part_of_speech(sentence_parse, maxlen=None):
    parts = sentence_parse.split('(')
    words = []
    parts_of_speech = []
    for p in parts:
        if ')' in p:
            res = p.split(' ')
            parts_of_speech.append(res[0])
            words.append(res[1].replace(')', ''))

    if maxlen is not None:
        words = words[:maxlen]
        parts_of_speech = parts_of_speech[:maxlen]
    return words, parts_of_speech


def get_all_snli_words(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        text = '[' + ','.join(lines) + ']'
        train = json.loads(text)

    words = []
    parts_of_speech = []
    for sample in tqdm(train):
        premise_words, premise_speech = get_words_with_part_of_speech(sample['sentence1_parse'])
        hypothesis_words, hypothesis_speech = get_words_with_part_of_speech(sample['sentence2_parse'])
        words += premise_words + hypothesis_words
        parts_of_speech += premise_speech + hypothesis_speech
    return words, parts_of_speech


def get_word_to_vecs(file_path, needed_words):
    res = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split(sep=' ')
            word = values[0]

            if word in needed_words:
                coefs = np.asarray(values[1:], dtype='float32')
                res[word] = coefs

    print('Found %s word vectors.' % len(res))
    return res


def get_word_to_ids(vectors_save_path='data/word-vectors.npy'):
    res = {}
    vectors = []

    for i, (word, vec) in enumerate(word_to_vec.items()):
        res[word] = i
        vectors.append(vec)

    vectors = np.array(vectors)
    np.save(vectors_save_path, vectors)
    return res


def label_to_one_hot(label, label_set=('entailment', 'contradiction', 'neutral')):
    res = np.zeros(shape=(len(label_set),), dtype=np.bool)
    for i, l in enumerate(label_set):
        if label == l:
            res[i] = 1
            break
    return res


def pad(x, maxlen):
    if len(x) <= maxlen:
        npad = ((0, maxlen - len(x)), (0, 0))
        return np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    res = x[:maxlen]
    return np.array(res)


def parse_one(premise_parse, hypothesis_parse, maxlen=32):
    # Words
    premise_words, premise_parts_of_speech = get_words_with_part_of_speech(premise_parse, maxlen=None)
    hypothesis_words, hypothesis_parts_of_speech = get_words_with_part_of_speech(hypothesis_parse, maxlen=None)
    premise_word_ids = [word_to_id[word] for word in premise_words]
    hypothesis_word_ids = [word_to_id[word] for word in hypothesis_words]

    # Syntactical features
    syntactical_premise = [part_of_speech_to_id[part] for part in premise_parts_of_speech]
    syntactical_hypothesis = [part_of_speech_to_id[part] for part in hypothesis_parts_of_speech]
    premise_hot = list(np.eye(len(part_of_speech_to_id) + 2)[syntactical_premise])  # Convert to 1-hot
    hypothesis_hot = list(np.eye(len(part_of_speech_to_id) + 2)[syntactical_hypothesis])  # Convert to 1-hot

    syntactical_premise = []
    syntactical_hypothesis = []
    for word, hot in zip(premise_words, premise_hot):
        l = list(hot)
        l.append(word in hypothesis_words)
        syntactical_premise.append(np.array(l))
    for word, hot in zip(hypothesis_words, hypothesis_hot):
        l = list(hot)
        l.append(word in premise_words)
        syntactical_hypothesis.append(np.array(l))
    syntactical_premise = np.array(syntactical_premise)
    syntactical_hypothesis = np.array(syntactical_hypothesis)

    # Chars
    premise_chars = []
    hypothesis_chars = []
    for word in premise_words:     premise_chars.append([char_to_id[c] for c in word])
    for word in hypothesis_words:  hypothesis_chars.append([char_to_id[c] for c in word])
    premise_chars = pad_sequences(premise_chars, maxlen=14, padding='post', truncating='post', value=0.)
    hypothesis_chars = pad_sequences(hypothesis_chars, maxlen=14, padding='post', truncating='post', value=0.)

    return np.array(premise_word_ids), pad(premise_chars, maxlen), pad(syntactical_premise, maxlen), \
           np.array(hypothesis_word_ids), pad(hypothesis_chars, maxlen), pad(syntactical_hypothesis, maxlen)


def save_train_data(directory, data):
    if not os.path.exists(directory):
        os.mkdir(directory)
    for i, item in tqdm(enumerate(data)):
        np.save(directory + '/' + str(i) + '.npy', item)


def load_train_data(directory):
    data = []
    for file in tqdm(os.listdir(directory)):
        if not file.endswith('.npy'):
            continue
        data.append( np.load(directory + '/' + file) )
    return data


def parse(data):
    input_word_p, input_word_h = [], []
    input_char_p, input_char_h = [], []
    input_syn_p,  input_syn_h = [], []
    labels = []

    for sample in tqdm(data):
        sample_inputs = parse_one(sample['sentence1_parse'], sample['sentence2_parse'])
        label = label_to_one_hot(sample['gold_label'])

        input_word_p.append(sample_inputs[0])
        input_char_p.append(sample_inputs[1])
        input_syn_p.append(sample_inputs[2])

        input_word_h.append(sample_inputs[3])
        input_char_h.append(sample_inputs[4])
        input_syn_h.append(sample_inputs[5])
        labels.append(label)

    input_word_p = pad_sequences(input_word_p, maxlen=32, padding='post', truncating='post', value=0.)
    input_word_h = pad_sequences(input_word_h, maxlen=32, padding='post', truncating='post', value=0.)
    input_char_p = np.array(input_char_p)
    input_char_h = np.array(input_char_h)
    input_syn_p = np.array(input_syn_p)
    input_syn_h = np.array(input_syn_h)
    labels = np.array(labels)

    return input_word_p, input_char_p, input_syn_p, \
           input_word_h, input_char_h, input_syn_h, \
           labels


if __name__ == '__main__':
    path = get_snli_file_path()
    train_w, train_p = get_all_snli_words(path + 'snli_1.0_train.jsonl')
    test_w, test_p = get_all_snli_words(path + 'snli_1.0_test.jsonl')
    dev_w, dev_p = get_all_snli_words(path + 'snli_1.0_dev.jsonl')

    words = train_w + test_w + dev_w
    parts_of_speech = train_p + test_p + dev_p
    words = set(words)
    parts_of_speech = set(parts_of_speech)

    print('Found', len(words), 'unique words')
    print('Found', len(parts_of_speech), 'unique parts of speech')

    word_to_vec = get_word_to_vecs(get_word2vec_file_path(), words)
    word_vector_size = len(word_to_vec['hello'])
    present_words = set(word_to_vec.keys())
    not_present_words = words - present_words

    print('#Present words:', len(present_words), '\t#Not present words', len(not_present_words))
    print('Inserting not present words to the list as uniform-random vectors')
    for word in not_present_words:
        word_to_vec[word] = np.random.uniform(size=word_vector_size)

    word_to_id = get_word_to_ids()
    chars = set()
    for word in word_to_id.keys():
        chars = chars.union(set(word))

    char_to_id = {}
    part_of_speech_to_id = {}
    for i, c in enumerate(chars):
        char_to_id[c] = i + 1
        print(c, end=' ')
    print('\n')

    for i, part in enumerate(parts_of_speech):
        part_of_speech_to_id[part] = i + 1
        print(part, end=' ')
    print('\n')
    print(len(char_to_id), len(word_to_id), len(part_of_speech_to_id))

    # Train
    data = load_data(path + 'snli_1.0_train.jsonl')
    save_train_data('data/train', parse(data))

    # Test
    data = load_data(path + 'snli_1.0_test.jsonl')
    save_train_data('data/test', parse(data))

    # Dev
    data = load_data(path + 'snli_1.0_dev.jsonl')
    save_train_data('data/dev', parse(data))
