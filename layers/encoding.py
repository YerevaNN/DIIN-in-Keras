from keras import backend as K
from keras.activations import softmax
from keras.engine.topology import Layer


class Encoding(Layer):
    def __init__(self, d, **kwargs):
        self.d = d
        self.w_itr_att = None
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        super(Encoding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_itr_att = self.add_weight(name='w_itr_att',
                                         shape=(3 * self.d),
                                         initializer='uniform',
                                         trainable=True)

        self.w1 = self.add_weight(name='W1',
                                  shape=(2 * self.d, self.d),
                                  initializer='uniform',
                                  trainable=True)
        self.w2 = self.add_weight(name='W2',
                                  shape=(2 * self.d, self.d),
                                  initializer='uniform',
                                  trainable=True)
        self.w3 = self.add_weight(name='W3',
                                  shape=(2 * self.d, self.d),
                                  initializer='uniform',
                                  trainable=True)

        self.b1 = self.add_weight(name='b1',
                                  shape=(self.d,),
                                  initializer='uniform',
                                  trainable=True)
        self.b2 = self.add_weight(name='b2',
                                  shape=(self.d,),
                                  initializer='uniform',
                                  trainable=True)
        self.b3 = self.add_weight(name='b3',
                                  shape=(self.d,),
                                  initializer='uniform',
                                  trainable=True)
        super(Encoding, self).build(input_shape)

    def alpha(self, a, b):
        return K.dot(K.transpose(self.w_itr_att),
                     K.concatenate([a, b, a * b]))

    def call(self, inputs_hw, **kwargs):
        l = K.int_shape(inputs_hw)[1]
        A = [self.alpha(inputs_hw[i], inputs_hw[j])
             for i in range(l)
             for j in range(l)]
        K.reshape(A, shape=(l, l))

        # Self-attention
        itr_attn = softmax(A, axis=2)  # Apply column-wise softmax
        # itr_attn = K.zeros(shape=(l,))
        # for i in range(l):
        #     for j in range(l):
        #         itr_attn[i] += (K.exp(A[i][j]) / K.sum(K.exp(A), axis=1)) * inputs_hw[j]

        # Fuse gate
        z = K.tanh(K.dot(K.transpose(self.w1),
                         K.concatenate([inputs_hw, itr_attn]))
                   + self.b1)
        r = K.sigmoid(K.dot(K.transpose(self.w2),
                            K.concatenate([inputs_hw, itr_attn]))
                      + self.b2)
        f = K.sigmoid(K.dot(K.transpose(self.w3),
                            K.concatenate([inputs_hw, itr_attn]))
                      + self.b3)

        encoding = r * inputs_hw + f * z
        return encoding

    def compute_output_shape(self, input_shape):
        # TODO
        pass
