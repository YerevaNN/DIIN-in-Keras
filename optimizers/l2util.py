from __future__ import division

import itertools

from keras import backend as K
from keras.optimizers import Optimizer


class BaseL2Optimizer(Optimizer):

    def __init__(self,
                 l2_full_step,
                 l2_full_ratio,
                 l2_difference_full_ratio,
                 **kwargs):
        super(BaseL2Optimizer, self).__init__(**kwargs)
        self.l2_full_step = K.variable(l2_full_step, name='l2_full_step')
        self.l2_full_ratio = K.variable(l2_full_ratio, name='l2_full_ratio')
        self.l2_difference_full_ratio = K.variable(l2_difference_full_ratio, name='l2_difference_full_ratio')

    def get_updates(self, loss, params):
        return NotImplementedError

    @staticmethod
    def compute_l2_ratio(time, l2_full_step, l2_full_ratio):
        return K.sigmoid((time - l2_full_step / 2) * 8 / (l2_full_step / 2)) * l2_full_ratio

    def add_decaying_l2_loss(self, loss, params, time):
        """ Add exponential decaying L2 regularization to all weights in the network """
        l2_ratio = self.compute_l2_ratio(time, self.l2_full_step, self.l2_full_ratio)
        for w in params:
            loss += K.sum(l2_ratio * K.square(w))

    def add_difference_l2_loss(self, loss, params):
        """ Add L2 regularization to penalize the difference between weights """
        penalize_difference = {}
        for param in params:
            if getattr(param, 'penalize_difference', None) is not None:
                weight_name = param.penalize_difference
                if weight_name not in penalize_difference:
                    penalize_difference[weight_name] = []
                penalize_difference[weight_name].append(param)

        for penalize_weights in penalize_difference.values():
            for w1, w2 in itertools.combinations(penalize_weights, 2):
                loss += K.sum(self.l2_difference_full_ratio * K.square(w1 - w2))

    def get_l2_loss(self, loss, params, iterations):
        iterations = K.cast(iterations, dtype='float32')
        self.add_decaying_l2_loss(loss, params, iterations)
        self.add_difference_l2_loss(loss, params)
        return loss

    def get_config(self):
        config = {'l2_full_step':               float(K.get_value(self.l2_full_step)),
                  'l2_full_ratio':              float(K.get_value(self.l2_full_ratio)),
                  'l2_difference_full_ratio':   float(K.get_value(self.l2_difference_full_ratio))}
        base_config = super(BaseL2Optimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
