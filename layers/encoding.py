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

    def call(self, P, **kwargs):
        # The paper takes inputs to be P as an example and then computes the same thing for H,
        # therefore we'll name our inputs P too.

        # Input of encoding is P with shape (batch, p, d). It would be (batch, h, d) for hypothesis
        # Construct alphaP of shape (batch, p, 3*d, p)
        # A = dot(w_itr_att, alphaP)

        # alphaPP consists of 3*d rows along 2nd axis
        # 1. up   -> first  d items represent P[i]
        # 2. mid  -> second d items represent P[j]
        # 3. down -> final items represent alpha(P[i], P[j]) which is element-wise product of P[i] and P[j] = P[i]*P[j]

        # If we look at one slice of alphaP we'll see that it has the following elements:
        # ----------------------------------------
        # P[i][0], P[i][0], P[i][0], ... P[i][0]   ▲
        # P[i][1], P[i][1], P[i][1], ... P[i][1]   |
        # P[i][2], P[i][2], P[i][2], ... P[i][2]   |
        # ...                              ...     | up
        #      ...                         ...     |
        #             ...                  ...     |
        # P[i][d], P[i][d], P[i][d], ... P[i][d]   ▼
        # ----------------------------------------
        # P[0][0], P[1][0], P[2][0], ... P[p][0]   ▲
        # P[0][1], P[1][1], P[2][1], ... P[p][1]   |
        # P[0][2], P[1][2], P[2][2], ... P[p][2]   |
        # ...                              ...     | mid
        #      ...                         ...     |
        #             ...                  ...     |
        # P[0][d], P[1][d], P[2][d], ... P[p][d]   ▼
        # ----------------------------------------
        #                                          ▲
        #                                          |
        #                                          |
        #               up * mid                   | down
        #          element-wise product            |
        #                                          |
        #                                          ▼
        # ----------------------------------------

        # For every slice(i) the up part changes its P[i] values
        # The middle part is repeated p times in depth (for every i)
        # So we can get the middle part by doing the following:
        # mid = broadcast(P.transpose, shape=(p, p, 3*d, p))
        # As we can notice up is the same mid, but with changed axis, so to obtain up from mid we can do:
        # up = swap_axes(mid, axis1=0, axis2=2)

        def broadcast_to(x, shape):
            return K.zeros(shape) + x

        batch, p, d = K.int_shape(P)
        P_transposed = K.transpose(P)
        mid = broadcast_to(P_transposed, shape=(p, d, p))
        up = broadcast_to(P_transposed, shape=(p, d, p))
        up = K.permute_dimensions(up, pattern=(2, 1, 0))
        down = up * mid

        alphaP = K.concatenate([up, mid, down], axis=2)
        A = K.dot(self.w_itr_att, alphaP)

        # Self-attention
        itr_attn = softmax(A, axis=2)  # Apply column-wise soft-max
        itr_attn = K.dot(itr_attn, P)

        # Fuse gate
        z = K.tanh(K.dot(K.transpose(self.w1),
                         K.concatenate([P, itr_attn]))
                   + self.b1)
        r = K.sigmoid(K.dot(K.transpose(self.w2),
                            K.concatenate([P, itr_attn]))
                      + self.b2)
        f = K.sigmoid(K.dot(K.transpose(self.w3),
                            K.concatenate([P, itr_attn]))
                      + self.b3)

        encoding = r * P + f * z
        return encoding

    def compute_output_shape(self, input_shape):
        # TODO
        pass
