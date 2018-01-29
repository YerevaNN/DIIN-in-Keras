from keras import backend as K
from keras.engine import Layer


class DecayingDropout(Layer):
    def __init__(self,
                 initial_keep_rate=1.,
                 decay_interval=10000,
                 decay_rate=0.977,
                 noise_shape=None,
                 seed=None,
                 **kwargs):

        super(DecayingDropout, self).__init__(**kwargs)
        self.iterations = None
        self.initial_keep_rate = initial_keep_rate
        self.decay_interval = decay_interval
        self.decay_rate = min(1., max(0., decay_rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def build(self, input_shape):
        self.iterations = self.add_weight(name='iterations', shape=(1,), dtype=K.floatx(),
                                          initializer='zeros', trainable=False)
        super(DecayingDropout, self).build(input_shape)

    def call(self, inputs, training=None):
        noise_shape = self._get_noise_shape(inputs)
        t = K.cast(self.iterations, K.floatx()) + 1
        p = t / float(self.decay_interval)

        keep_rate = self.initial_keep_rate * K.pow(self.decay_rate, p)

        def dropped_inputs():
            self.add_update([K.update_add(self.iterations, [1])], inputs)
            return K.dropout(inputs, 1 - keep_rate[0], noise_shape, seed=self.seed)
        return K.in_train_phase(dropped_inputs, inputs, training=training)

    def get_config(self):
        config = {'initial_keep_rate':  self.initial_keep_rate,
                  'decay_interval':     self.decay_interval,
                  'decay_rate':         self.decay_rate,
                  'noise_shape':        self.noise_shape,
                  'seed':               self.seed}
        base_config = super(DecayingDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
