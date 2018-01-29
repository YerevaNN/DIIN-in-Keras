"""DenseNet models for Keras.
    :ref  [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
          [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation]
            (https://arxiv.org/pdf/1611.09326.pdf)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine import Model
from keras.layers import Input, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.pooling import MaxPooling2D


class DenseNet(Model):
    def __init__(self, input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_layers_per_block=-1,
                 compression=1.0, dropout_rate=0., input_tensor=None, apply_batch_norm=False,
                 include_top=True, classes=10, activation='softmax', name='DenseNet', inputs=None, outputs=None):
        """Instantiate the DenseNet architecture,
            Note that when using TensorFlow,
            for best performance you should set
            `image_data_format='channels_last'` in your Keras config
            at ~/.keras/keras.json.
            The model and the weights are compatible with both
            TensorFlow and Theano. The dimension ordering
            convention used by the model is the one
            specified in your Keras config file.
            # Arguments
                input_shape: optional shape tuple, only to be specified
                    if `include_top` is False (otherwise the input shape
                    has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                    or `(3, 32, 32)` (with `channels_first` dim ordering).
                    It should have exactly 3 inputs channels,
                    and width and height should be no smaller than 8.
                    E.g. `(200, 200, 3)` would be one valid value.
                depth: number or layers in the DenseNet
                nb_dense_block: number of dense blocks to add to end (generally = 3)
                growth_rate: number of filters to add per dense block
                nb_filter: initial number of filters. -1 indicates initial
                    number of filters is 2 * growth_rate
                nb_layers_per_block: number of layers in each dense block.
                    Can be a -1, positive integer or a list.
                    If -1, calculates nb_layer_per_block from the network depth.
                    If positive integer, a set number of layers per dense block.
                    If list, nb_layer is used as provided. Note that list size must
                    be (nb_dense_block + 1)
                bottleneck: flag to add bottleneck blocks in between dense blocks
                compression: scale down ratio of feature maps.
                dropout_rate: dropout rate
                include_top: whether to include the fully-connected
                    layer at the top of the network.
                input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                    to use as image input for the model.
                classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
                activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
            # Returns
                A Keras model instance.
            """

        if inputs or outputs:
            super(DenseNet, self).__init__(inputs, outputs, name=name)
            return

        if activation not in ['softmax', 'sigmoid']:
            raise ValueError('activation must be one of "softmax" or "sigmoid"')

        if activation == 'sigmoid' and classes != 1:
            raise ValueError('sigmoid activation can only be used when classes = 1')

        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=32,
                                          min_size=8,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top)

        if input_tensor is None:
            net_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                net_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                net_input = input_tensor

        # Construct DenseNet
        self.concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        # Compute number of layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            assert len(nb_layers) == nb_dense_block, 'If list, nb_layer is used as provided. ' \
                                                     'Note that list size must be (nb_dense_block)'
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
                count = int((depth - 4) / 3)
                nb_layers = [count for _ in range(nb_dense_block)]
            else:
                nb_layers = [nb_layers_per_block] * nb_dense_block

        # Add dense blocks
        x = net_input
        for block_idx in range(nb_dense_block):
            x, nb_filter = self.__dense_block(x, nb_layers[block_idx], growth_rate, dropout_rate, apply_batch_norm)
            x = self.__transition_block(x, nb_filter, compression=compression, apply_batch_norm=apply_batch_norm)

        if apply_batch_norm:
            x = BatchNormalization(axis=self.concat_axis, epsilon=1.1e-5)(x)

        # Flatten if the shapes are known otherwise apply average pooling
        try:    x = Flatten()(x)
        except: x = GlobalAveragePooling2D()(x)

        if include_top:
            x = Dense(classes, activation=activation)(x)

        super(DenseNet, self).__init__(inputs=net_input, outputs=x, name=name)

    def __conv_block(self, x, nb_filter, dropout_rate=None, apply_batch_norm=False):
        """ Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
        Args:
            x: Input keras tensor
            nb_filter: number of filters
            dropout_rate: dropout rate
        Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
        """

        if apply_batch_norm:
            x = BatchNormalization(axis=self.concat_axis, epsilon=1.1e-5)(x)

        x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu')(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x

    def __dense_block(self, x, nb_layers, growth_rate, dropout_rate=None, apply_batch_norm=False):
        """
        Build a dense_block where the output of each conv_block is fed to subsequent ones
        :return x, number of filters output layer has
        """
        for i in range(nb_layers):
            cb = self.__conv_block(x, growth_rate, dropout_rate, apply_batch_norm=apply_batch_norm)
            x = concatenate([x, cb], axis=self.concat_axis)
        return x, K.int_shape(x)[self.concat_axis]

    def __transition_block(self, x, nb_filter, compression, apply_batch_norm):
        if apply_batch_norm:
            x = BatchNormalization(axis=self.concat_axis, epsilon=1.1e-5)(x)
        x = Conv2D(int(nb_filter * compression), (1, 1), padding='same', activation=None)(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        return x
