import keras.backend as K
from keras.engine import Layer


class Interaction(Layer):

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        premise_encoding, hypothesis_encoding = inputs

        # Perform element-wise multiplication for each row of premise and hypothesis
        # For every i, j premise_row[i] * hypothesis_row[j]
        # betta(premise_encoding[i], hypothesis_encoding[j]) = premise_encoding[i] * hypothesis_encoding[j]

        # => we can do the following:
        # 1. broadcast premise    to shape (batch, p, h, d)
        # 2. broadcast hypothesis to shape (batch, p, h, d)
        # perform premise * hypothesis

        # In keras this operation is equivalent to reshaping premise (batch, p, 1, d), hypothesis (batch, 1, h, d)
        # And then compute premise * hypothesis
        premise_encoding = K.expand_dims(premise_encoding, axis=2)          # (batch, p, 1, d)
        hypothesis_encoding = K.expand_dims(hypothesis_encoding, axis=1)    # (batch, 1, h, d)

        # Compute interaction tensor I = betta(premise, hypothesis)
        I = premise_encoding * hypothesis_encoding
        return I

    def compute_output_shape(self, input_shape):
        premise_shape = input_shape[0]
        hypothesis_shape = input_shape[1]

        # (batch, p, d), (batch, h, d) => (batch, p, h, d)
        assert len(premise_shape) == len(hypothesis_shape) == 3
        assert premise_shape[0] == hypothesis_shape[0]
        assert premise_shape[2] == hypothesis_shape[2]
        batch = premise_shape[0]
        p = premise_shape[1]
        h = hypothesis_shape[1]
        d = hypothesis_shape[2]

        return batch, p, h, d
