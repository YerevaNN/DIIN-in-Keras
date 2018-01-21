from keras import backend as K
from keras.engine import Layer


class DecayingDropout(Layer):
    def __init__(self,
                 initial_keep_rate=1,
                 decay_interval=10000,
                 decay_rate=0.977,
                 noise_shape=None,
                 seed=None,
                 **kwargs):

        super(DecayingDropout, self).__init__(**kwargs)
        self.current_keep_rate = None
        self.current_step = None
        self.initial_keep_rate = initial_keep_rate
        self.decay_interval = decay_interval
        self.decay_rate = min(1., max(0., decay_rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def build(self, input_shape):
        self.current_keep_rate = K.variable(self.initial_keep_rate, name='keep_rate')
        self.current_step = K.variable(1, name='time')
        super(DecayingDropout, self).build(input_shape)

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        noise_shape = self._get_noise_shape(inputs)
        one = K.ones(shape=K.int_shape(self.current_step))

        def dropped_inputs():
            return K.dropout(inputs, 1 - keep_rate, noise_shape, seed=self.seed)

        keep_rate = self.initial_keep_rate * K.pow(self.decay_rate, self.current_step / self.decay_interval)
        self.add_update([K.update_add(self.current_step, one)], inputs)
        return K.in_train_phase(dropped_inputs, inputs, training=training)
