# -*- coding: utf-8 -*-
from keras import backend as K
from keras.activations import softmax
from keras.engine.topology import Layer

from layers.decaying_dropout import DecayingDropout
from util import broadcast_last_axis


class Encoding(Layer):
    def __init__(self, **kwargs):
        self.w_itr_att = None
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        super(Encoding, self).__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]
        self.w_itr_att = self.add_weight(name='w_itr_att', shape=(3 * d,), initializer='glorot_uniform')

        self.w1 = self.add_weight(name='W1', shape=(2 * d, d,), initializer='glorot_uniform')
        self.w2 = self.add_weight(name='W2', shape=(2 * d, d,), initializer='glorot_uniform')
        self.w3 = self.add_weight(name='W3', shape=(2 * d, d,), initializer='glorot_uniform')

        self.b1 = self.add_weight(name='b1', shape=(d,), initializer='zeros')
        self.b2 = self.add_weight(name='b2', shape=(d,), initializer='zeros')
        self.b3 = self.add_weight(name='b3', shape=(d,), initializer='zeros')

        # Add parameters for weights to penalize difference between them
        # Optimizer will penalize weight difference between all occurrences of the same name
        self.w_itr_att.penalize_difference = 'w_itr_attn'
        self.w1.penalize_difference = 'w1'
        self.w2.penalize_difference = 'w2'
        self.w3.penalize_difference = 'w3'
        self.b1.penalize_difference = 'b1'
        self.b2.penalize_difference = 'b2'
        self.b3.penalize_difference = 'b3'
        super(Encoding, self).build(input_shape)

    def call(self, P, **kwargs):
        """
        :param P: inputs
        :return: encoding of inputs P
        """
        ''' Paper notations in the code '''
        # P = P_hw
        # itr_attn = P_itrAtt
        # encoding = P_enc
        # The paper takes inputs to be P(_hw) as an example and then computes the same thing for H,
        # therefore we'll name our inputs P too.

        # Input of encoding is P with shape (batch, p, d). It would be (batch, h, d) for hypothesis
        # Construct alphaP of shape (batch, p, 3*d, p)
        # A = dot(w_itr_att, alphaP)

        # alphaP consists of 3*d rows along 2nd axis
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
        # mid = broadcast(P) -> to get tensor of shape (batch, p, d, p)
        # As we can notice up is the same mid, but with changed axis, so to obtain up from mid we can do:
        # up = swap_axes(mid, axis1=0, axis2=2)

        ''' Alpha '''
        # P                                                     # (batch, p, d)
        mid = broadcast_last_axis(P)                            # (batch, p, d, p)
        up = K.permute_dimensions(mid, pattern=(0, 3, 2, 1))    # (batch, p, d, p)
        alphaP = K.concatenate([up, mid, up * mid], axis=2)     # (batch, p, 3d, p)
        A = K.dot(self.w_itr_att, alphaP)                       # (batch, p, p)

        ''' Self-attention '''
        # P_itr_attn[i] = sum of for j = 1...p:
        #                           s = sum(for k = 1...p:  e^A[k][j]
        #                           ( e^A[i][j] / s ) * P[j]  --> P[j] is the j-th row, while the first part is a number
        # So P_itr_attn is the weighted sum of P
        # SA is column-wise soft-max applied on A
        # P_itr_attn[i] is the sum of all rows of P scaled by i-th row of SA
        SA = softmax(A, axis=2)        # (batch, p, p)
        itr_attn = K.batch_dot(SA, P)  # (batch, p, d)

        ''' Fuse gate '''
        # These layers are considered linear in the official implementation, therefore we apply dropout on each input
        P_concat = K.concatenate([P, itr_attn], axis=2)                         # (batch, p, 2d)
        z = K.tanh(K.dot(DecayingDropout()(P_concat), self.w1) + self.b1)       # (batch, p, d)
        r = K.sigmoid(K.dot(DecayingDropout()(P_concat), self.w2) + self.b2)    # (batch, p, d)
        f = K.sigmoid(K.dot(DecayingDropout()(P_concat), self.w3) + self.b3)    # (batch, p, d)

        encoding = r * P + f * z        # (batch, p, d)
        return encoding                 # (batch, p, d)
