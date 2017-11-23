import keras.backend as K
from keras.engine import Layer


class Interaction(Layer):

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        premise_encoding = inputs[0]
        hypothesis_encoding = inputs[1]

        # Perform element-wise multiplication for each row of premise and hypothesis
        # betta(premise_encoding[i], hypothesis_encoding[j]) = premise_encoding[i] * hypothesis_encoding[j]
        return K.dot(premise_encoding, hypothesis_encoding)

    def compute_output_shape(self, input_shape):
        premise_shape = input_shape[0]
        hypothesis_shape = input_shape[1]

        # (batch, p, d), (batch, h, d) => (batch, p, h, d)
        assert premise_shape[2] == hypothesis_shape[2]
        batch = premise_shape[0]
        p = premise_shape[1]
        h = premise_shape[1]
        d = premise_shape[2]

        return batch, p, h, d
