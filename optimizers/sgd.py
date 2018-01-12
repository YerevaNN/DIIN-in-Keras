from keras.optimizers import SGD

from optimizers.l2util import BaseL2Optimizer


class SGDL2(BaseL2Optimizer, SGD):
    def __init__(self,
                 l2_full_step=100000.,
                 l2_full_ratio=9e-5,
                 l2_difference_full_ratio=1e-3,
                 **kwargs):
        super(SGDL2, self).__init__(l2_full_step=l2_full_step,
                                    l2_full_ratio=l2_full_ratio,
                                    l2_difference_full_ratio=l2_difference_full_ratio,
                                    **kwargs)

    def get_updates(self, loss, params):
        loss = BaseL2Optimizer.get_l2_loss(self, loss=loss, params=params, iterations=self.iterations)
        return SGD.get_updates(self, loss=loss, params=params)
