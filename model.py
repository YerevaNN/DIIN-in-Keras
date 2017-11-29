from keras.engine import Model
from keras.layers import Input, Embedding, Conv1D, MaxPool1D, concatenate, Flatten, Dense, Conv2D

from densenet.densenet import get_densenet_output
from layers.encoding import Encoding
from layers.interaction import Interaction


def construct_model(p=None, h=None, d=300, embedding_size=30, word_embedding_size=300, FSDR=0.3, TSDR=0.5, GR=20, n=8):

    """
    :param p: sequence length of premise
    :param h: sequence length of hypothesis
    :param d: the dimension of both premise and hypothesis representations
    :param word_embedding_size: size of the word-embedding vector (default GloVe is 300)
    :param embedding_size: input size of the character-embedding layer
    :param FSDR: first scale down ratio in densenet
    :param TSDR: transition scale down ratio in densenet
    :param GR: growing rate in densenet
    :param n: number of layers in one dense-block

    :ref https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-
    :returns DIIN model
    """

    '''Embedding layer'''
    # Word embeddings + char-level embeddings + features

    # 1. Word embedding input
    word_embedding_size = d
    premise_word_input = Input(shape=(p, word_embedding_size), name='PremiseWordInput')
    hypothesis_word_input = Input(shape=(h, word_embedding_size), name='HypothesisWordInput')

    # # 2. Character input
    # premise_char_input = Input(shape=(embedding_size,))
    # hypothesis_char_input = Input(shape=(embedding_size,))
    #
    # # Share weights of character-level embedding for premise and hypothesis
    # character_embedding_layer = Embedding(embedding_size, output_dim=d)
    # premise_embedding = character_embedding_layer(premise_char_input)
    # hypothesis_embedding = character_embedding_layer(hypothesis_char_input)
    #
    # # Share weights of 1D convolution for premise and hypothesis
    # character_conv_layer = Conv1D(filters=d, kernel_size=15)
    # premise_embedding = character_conv_layer(premise_embedding)
    # hypothesis_embedding = character_conv_layer(hypothesis_embedding)
    #
    # # Apply max-pooling after convolution
    # premise_embedding = MaxPool1D(premise_embedding)
    # hypothesis_embedding = MaxPool1D(hypothesis_embedding)
    #
    # # Concatenate all features
    # premise_embedding = concatenate([premise_word_input, premise_embedding])
    # hypothesis_embedding = concatenate([hypothesis_word_input, hypothesis_embedding])
    premise_embedding = premise_word_input
    hypothesis_embedding = hypothesis_word_input

    '''Encoding layer'''
    # --Now we have the embedded premise [pxd] along with embedded hypothesis [hxd]--
    premise_encoding = Encoding(d=d, name='PremiseEncoding')(premise_embedding)
    hypothesis_encoding = Encoding(d=d, name='HypothesisEncoding')(hypothesis_embedding)

    '''Interaction layer'''
    interaction = Interaction(name='Interaction')([premise_encoding, hypothesis_encoding])

    '''Feature Extraction layer'''
    feature_layers = int(d * FSDR)
    feature_extractor_input = Conv2D(feature_layers, kernel_size=1, activation=None)(interaction)
    feature_extractor = get_densenet_output(include_top=False,
                                            weights=None,
                                            input_tensor=feature_extractor_input,
                                            nb_dense_block=3,
                                            nb_layers_per_block=n,
                                            compression=TSDR,
                                            growth_rate=GR)

    '''Output layer'''
    out = Dense(3, activation='softmax', name='Output')(feature_extractor)

    return Model(inputs=[premise_word_input, hypothesis_word_input],
                 outputs=out,
                 name='DIIN')
