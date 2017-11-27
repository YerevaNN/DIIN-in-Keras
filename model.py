from keras.engine import Model
from keras.layers import Input, Embedding, Conv1D, MaxPool1D, concatenate, Flatten, Dense

from densenet.densenet import DenseNet
from layers.encoding import Encoding
from layers.interaction import Interaction


def construct_model(p=None, h=None, d=300, embedding_size=30, word_embedding_size=300, FSDR=0.3, TSDR=0.5, GR=20):

    """
    :param word_embedding_size: size of the word-embedding vector (default GloVe is 300)
    :param embedding_size: input size of the character-embedding layer
    :param p: sequence length of premise
    :param h: sequence length of hypothesis
    :param d: the dimension of both premise and hypothesis representations

    :ref https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-
    :returns DIIN model
    """

    '''Embedding layer'''
    print('p:', p, 'h:', h, 'd:', d)
    # Word embeddings + char-level embeddings + features

    # 1. Word embedding input
    word_embedding_size = d
    premise_word_input = Input(shape=(p, word_embedding_size))
    hypothesis_word_input = Input(shape=(h, word_embedding_size))

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
    premise_encoding = Encoding(d=d)(premise_embedding)
    hypothesis_encoding = Encoding(d=d)(hypothesis_embedding)

    '''Interaction layer'''
    interaction = Interaction()([premise_encoding, hypothesis_encoding])

    '''Feature Extraction layer'''
    feature_extractor = DenseNet(include_top=False, weights=None, input_tensor=interaction, growth_rate=GR)(interaction)
    features = Flatten()(feature_extractor)
    out = Dense(3, activation='softmax')(features)

    return Model(inputs=[premise_word_input, hypothesis_word_input],
                 outputs=out,
                 name='DIIN')
