from keras import backend as K
from keras.engine import Layer
from keras.initializers import Constant


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
        self.current_keep_rate = self.add_weight(shape=(1,),
                                                 name='keep_rate',
                                                 initializer=Constant(value=self.initial_keep_rate),
                                                 trainable=False)
        self.current_step = self.add_weight(shape=(1,),
                                            name='step',
                                            initializer=Constant(value=0),
                                            trainable=False)
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
        zero = K.zeros(shape=K.int_shape(self.current_step))

        def dropped_inputs():
            return K.dropout(inputs, 1 - self.current_keep_rate[0], noise_shape, seed=self.seed)

        new_keep_rate = K.switch(K.all(self.current_step % self.decay_interval == zero),
                                 self.current_keep_rate * self.decay_rate,
                                 self.current_keep_rate)

        self.add_update([K.update_add(self.current_step, one),
                         K.update(self.current_keep_rate, new_keep_rate)],
                        inputs)
        return K.in_train_phase(dropped_inputs, inputs, training=training)
