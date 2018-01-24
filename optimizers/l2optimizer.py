from __future__ import division

import itertools

from keras import backend as K
from keras.optimizers import Optimizer, serialize, deserialize


class L2Optimizer(Optimizer):

    def __init__(self,
                 optimizer,
                 l2_full_step=100000.,
                 l2_full_ratio=9e-5,
                 l2_difference_full_ratio=1e-3):
        """
        :param optimizer: real optimizer which is wrapped by L2Optimizer
        :param l2_full_step: determines at which step the maximum L2 regularization ratio would be applied
        :param l2_full_ratio: determines the maximum L2 regularization ratio
        :param l2_difference_full_ratio: determines the maximum L2 regularization for weight difference penalization
        """
        super(L2Optimizer, self).__init__()
        self.optimizer = optimizer
        self.l2_full_step = K.variable(l2_full_step, name='l2_full_step')
        self.l2_full_ratio = K.variable(l2_full_ratio, name='l2_full_ratio')
        self.l2_difference_full_ratio = K.variable(l2_difference_full_ratio, name='l2_difference_full_ratio')

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
        t = K.cast(iterations, dtype=K.floatx())
        self.add_decaying_l2_loss(loss, params, t)
        self.add_difference_l2_loss(loss, params)
        return loss

    def get_updates(self, loss, params):
        loss = self.get_l2_loss(loss=loss, params=params, iterations=self.optimizer.iterations)
        return self.optimizer.get_updates(loss=loss, params=params)

    def get_config(self):
        config = {'optimizer':                  serialize(self.optimizer),
                  'l2_full_step':               float(K.get_value(self.l2_full_step)),
                  'l2_full_ratio':              float(K.get_value(self.l2_full_ratio)),
                  'l2_difference_full_ratio':   float(K.get_value(self.l2_difference_full_ratio))}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer_config = config.pop('optimizer')
        optimizer = deserialize(optimizer_config)
        return cls(optimizer=optimizer, **config)

    def set_weights(self, weights):
        self.optimizer.set_weights(weights)

    def get_weights(self):
        return self.optimizer.get_weights()
