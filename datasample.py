import numpy as np


class Sample:
    def __init__(self, premise, premise_word_vector, hypothesis, hypothesis_word_vector, label):
        self.premise = premise
        self.premise_word_vector = np.array(premise_word_vector)

        self.hypothesis = hypothesis
        self.hypothesis_word_vector = np.array(hypothesis_word_vector)

        self.label = label
        self.label_vector = self.label_to_one_hot()

    def label_to_one_hot(self, label_set=('entailment', 'contradiction', 'neutral')):
        res = np.zeros(shape=(len(label_set),), dtype=np.bool)
        for i, l in enumerate(label_set):
            if self.label == l:
                res[i] = 1
                break
        return res

    def to_data_sample(self):
        """
        :return: list(input1, input2....), label
        """
        return [self.premise_word_vector, self.hypothesis_word_vector], self.label_vector

    def from_numpy_array(self, arr):
        self.premise_word_vector = arr[0]
        self.hypothesis_word_vector = arr[1]
        self.label_vector = arr[2]

    def __eq__(self, other):
        assert isinstance(other, Sample)
        return (self.premise_word_vector.shape == other.premise_word_vector.shape) and \
               (self.hypothesis_word_vector.shape == other.hypothesis_word_vector.shape)

    def __lt__(self, other):
        assert isinstance(other, Sample)
        return (self.premise_word_vector.shape,  self.hypothesis_word_vector.shape) < \
               (other.premise_word_vector.shape, other.hypothesis_word_vector.shape)

    def __str__(self):
        return 'Premise: ' + str(self.premise_word_vector.shape) + \
               '\nHypothesis: ' + str(self.hypothesis_word_vector.shape) + \
               '\nlabel: ' + str(self.label) + '\n'
