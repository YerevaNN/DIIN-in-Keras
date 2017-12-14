import numpy as np
from keras import backend as K
from keras.engine import Model
from keras.layers import Input, Dense, Conv2D, Embedding, Conv1D, TimeDistributed, GlobalMaxPooling1D, Concatenate
from keras.models import Sequential

from feature_extractors.densenet import get_densenet_output
from layers.decaying_dropout import DecayingDropout
from layers.encoding import Encoding
from layers.interaction import Interaction


def construct_model(p=None,
                    h=None,
                    word_embedding_weights=np.array([]),
                    word_embedding_size=300,
                    char_embedding_size=30,
                    char_pad_size=14,
                    syntactical_feature_size=7,
                    FSDR=0.3,
                    TSDR=0.5,
                    GR=20,
                    n=8):

    """
    :param p: sequence length of premise
    :param h: sequence length of hypothesis
    :param word_embedding_weights: matrix of weights for word embeddings (GloVe pre-trained vectors)
    :param word_embedding_size: size of the word-embedding vector (default GloVe is 300)
    :param char_embedding_size: input size of the character-embedding layer
    :param char_pad_size: length of the padding size for each word
    :param syntactical_feature_size: size of the syntactical feature vector for each word
    :param FSDR: first scale down ratio in densenet
    :param TSDR: transition scale down ratio in densenet
    :param GR: growing rate in densenet
    :param n: number of layers in one dense-block

    :ref https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-
    :returns DIIN model
    """

    '''Embedding layer'''
    # 1. Word embedding input
    premise_word_input = Input(shape=(p,),    name='PremiseWordInput',    dtype='int64')
    hypothesis_word_input = Input(shape=(h,), name='HypothesisWordInput', dtype='int64')

    word_embedding = Embedding(input_dim=word_embedding_weights.shape[0],
                               output_dim=word_embedding_size,
                               weights=[word_embedding_weights],
                               trainable=True,
                               name='WordEmbedding')
    premise_word_embedding = word_embedding(premise_word_input)
    hypothesis_word_embedding = word_embedding(hypothesis_word_input)
    premise_word_embedding = DecayingDropout(decay_interval=10000, decay_rate=0.977)(premise_word_embedding)
    hypothesis_word_embedding = DecayingDropout(decay_interval=10000, decay_rate=0.977)(hypothesis_word_embedding)

    # 2. Character input
    premise_char_input = Input(shape=(p, char_pad_size,))
    hypothesis_char_input = Input(shape=(h, char_pad_size,))

    # Share weights of character-level embedding for premise and hypothesis
    character_embedding_layer = TimeDistributed(Sequential([
        Embedding(input_dim=128, output_dim=char_embedding_size, input_length=char_pad_size),
        Conv1D(filters=40, kernel_size=3),
        GlobalMaxPooling1D()
    ]))
    character_embedding_layer.build(input_shape=(None, None, char_pad_size))
    premise_char_embedding = character_embedding_layer(premise_char_input)
    hypothesis_char_embedding = character_embedding_layer(hypothesis_char_input)

    # 3. Syntactical features
    premise_syntactical_input = Input(shape=(p, syntactical_feature_size,))
    hypothesis_syntactical_input = Input(shape=(h, syntactical_feature_size,))

    # Concatenate all features
    premise_embedding = Concatenate()([premise_word_embedding,       premise_char_embedding,    premise_syntactical_input])
    hypothesis_embedding = Concatenate()([hypothesis_word_embedding, hypothesis_char_embedding, hypothesis_syntactical_input])
    d = K.int_shape(hypothesis_embedding)[-1]

    '''Encoding layer'''
    # --Now we have the embedded premise [pxd] along with embedded hypothesis [hxd]--
    premise_encoding = Encoding(d=d,    name='PremiseEncoding')(premise_embedding)
    hypothesis_encoding = Encoding(d=d, name='HypothesisEncoding')(hypothesis_embedding)

    '''Interaction layer'''
    interaction = Interaction(name='Interaction')([premise_encoding, hypothesis_encoding])

    '''Feature Extraction layer'''
    feature_layers = int(d * FSDR)
    feature_extractor_input = Conv2D(filters=feature_layers, kernel_size=1, activation=None)(interaction)
    feature_extractor = get_densenet_output(include_top=False,
                                            weights=None,
                                            input_tensor=feature_extractor_input,
                                            nb_dense_block=3,
                                            nb_layers_per_block=n,
                                            compression=TSDR,
                                            growth_rate=GR)

    '''Output layer'''
    features = DecayingDropout(decay_interval=10000, decay_rate=0.977)(feature_extractor)
    out = Dense(units=3, activation='softmax', name='Output')(features)

    return Model(inputs=[premise_word_input,    premise_char_input,    premise_syntactical_input,
                         hypothesis_word_input, hypothesis_char_input, hypothesis_syntactical_input],
                 outputs=out,
                 name='DIIN')
