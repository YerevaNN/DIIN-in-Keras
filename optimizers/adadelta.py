from keras.optimizers import Adadelta

from optimizers.l2util import BaseL2Optimizer, compute_decaying_l2_loss, compute_decaying_difference_l2_loss


class AdadeltaL2(Adadelta, BaseL2Optimizer):
    def __init__(self,
                 l2_full_step=100000.,
                 l2_full_ratio=0.9e-5,
                 l2_difference_full_ratio=1e-3,
                 **kwargs):
        Adadelta.__init__(self, **kwargs)
        BaseL2Optimizer.__init__(self,
                                 l2_full_step=l2_full_step,
                                 l2_full_ratio=l2_full_ratio,
                                 l2_difference_full_ratio=l2_difference_full_ratio,
                                 **kwargs)

    def get_updates(self, loss, params):
        loss = compute_decaying_l2_loss(loss, params, self.iterations, self.l2_full_step, self.l2_full_ratio)
        loss = compute_decaying_difference_l2_loss(loss, params, self.iterations, self.l2_full_step, self.l2_difference_full_ratio)

        return super(AdadeltaL2, self).get_updates(loss, params)
