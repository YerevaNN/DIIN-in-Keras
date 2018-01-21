from keras import backend as K
from keras.engine import Model
from keras.layers import Input, Dense, Conv2D, Embedding, Conv1D, TimeDistributed, GlobalMaxPooling1D, Concatenate
from keras.models import Sequential

from feature_extractors.densenet import DenseNet
from layers.decaying_dropout import DecayingDropout
from layers.encoding import Encoding
from layers.interaction import Interaction


class DIIN(Model):
    def __init__(self,
                 p,
                 h,
                 word_embedding_weights,
                 char_embedding_size=30,
                 chars_per_word=14,
                 syntactical_feature_size=50,
                 dropout_decay_interval=10000,
                 dropout_decay_rate=0.977,
                 dropout_initial_keep_rate=1.,
                 char_conv_filters=77,
                 char_conv_kernel_size=5,
                 FSDR=0.3,
                 TSDR=0.5,
                 GR=20,
                 n=8,
                 nb_dense_blocks=3,
                 nb_labels=3,
                 name='DIIN'):
        """
        :ref https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-

        :param p: sequence length of premise
        :param h: sequence length of hypothesis
        :param word_embedding_weights: matrix of weights for word embeddings (GloVe pre-trained vectors)
        :param char_embedding_size: input size of the character-embedding layer
        :param chars_per_word: length of the padding size for each word
        :param syntactical_feature_size: size of the syntactical feature vector for each word
        :param dropout_initial_keep_rate: initial state of dropout
        :param dropout_decay_rate: how much to change dropout at each interval
        :param dropout_decay_interval: how much time to wait for the next update
        :param char_conv_filters: number of conv-filters applied on character embedding
        :param char_conv_kernel_size: size of the kernel applied on character embeddings
        :param FSDR: first scale down ratio in densenet
        :param TSDR: transition scale down ratio in densenet
        :param GR: growing rate in densenet
        :param n: number of layers in one dense-block
        :param nb_dense_blocks: number of dense blocks in densenet
        :param nb_labels: number of labels
        """

        '''Embedding layer'''
        # 1. Word embedding input
        premise_word_input    = Input(shape=(p,), dtype='int64', name='PremiseWordInput')
        hypothesis_word_input = Input(shape=(h,), dtype='int64', name='HypothesisWordInput')

        print(word_embedding_weights.shape)

        word_embedding = Embedding(input_dim=word_embedding_weights.shape[0],
                                   output_dim=word_embedding_weights.shape[1],
                                   weights=[word_embedding_weights],
                                   trainable=True,
                                   name='WordEmbedding')
        premise_word_embedding    = word_embedding(premise_word_input)
        hypothesis_word_embedding = word_embedding(hypothesis_word_input)

        premise_word_embedding    = DecayingDropout(initial_keep_rate=dropout_initial_keep_rate,
                                                    decay_interval=dropout_decay_interval,
                                                    decay_rate=dropout_decay_rate,
                                                    name='PremiseWordEmbeddingDropout')(premise_word_embedding)
        hypothesis_word_embedding = DecayingDropout(initial_keep_rate=dropout_initial_keep_rate,
                                                    decay_interval=dropout_decay_interval,
                                                    decay_rate=dropout_decay_rate,
                                                    name='HypothesisWordEmbeddingDropout')(hypothesis_word_embedding)

        # 2. Character input
        premise_char_input    = Input(shape=(p, chars_per_word,), name='PremiseCharInput')
        hypothesis_char_input = Input(shape=(h, chars_per_word,), name='HypothesisCharInput')

        # Share weights of character-level embedding for premise and hypothesis
        character_embedding_layer = TimeDistributed(Sequential([
            Embedding(input_dim=128, output_dim=char_embedding_size, input_length=chars_per_word),
            Conv1D(filters=char_conv_filters, kernel_size=char_conv_kernel_size),
            GlobalMaxPooling1D()
        ]), name='CharEmbedding')
        character_embedding_layer.build(input_shape=(None, None, chars_per_word))
        premise_char_embedding    = character_embedding_layer(premise_char_input)
        hypothesis_char_embedding = character_embedding_layer(hypothesis_char_input)

        premise_char_embedding = DecayingDropout(initial_keep_rate=dropout_initial_keep_rate,
                                                 decay_interval=dropout_decay_interval,
                                                 decay_rate=dropout_decay_rate,
                                                 name='PremiseCharEmbedding')(premise_char_embedding)
        hypothesis_char_embedding = DecayingDropout(initial_keep_rate=dropout_initial_keep_rate,
                                                    decay_interval=dropout_decay_interval,
                                                    decay_rate=dropout_decay_rate,
                                                    name='HypothesisCharEmbedding')(hypothesis_char_embedding)

        # 3. Syntactical features
        premise_syntactical_input    = Input(shape=(p, syntactical_feature_size,), name='PremiseSyntacticalInput')
        hypothesis_syntactical_input = Input(shape=(h, syntactical_feature_size,), name='HypothesisSyntacticalInput')

        # Concatenate all features
        premise_embedding    = Concatenate(name='PremiseEmbedding')(   [premise_word_embedding,    premise_char_embedding,    premise_syntactical_input])
        hypothesis_embedding = Concatenate(name='HypothesisEmbedding')([hypothesis_word_embedding, hypothesis_char_embedding, hypothesis_syntactical_input])
        d = K.int_shape(hypothesis_embedding)[-1]

        '''Encoding layer'''
        # Now we have the embedded premise [pxd] along with embedded hypothesis [hxd]
        premise_encoding    = Encoding(d=d, name='PremiseEncoding')(premise_embedding)
        hypothesis_encoding = Encoding(d=d, name='HypothesisEncoding')(hypothesis_embedding)

        '''Interaction layer'''
        interaction = Interaction(name='Interaction')([premise_encoding, hypothesis_encoding])
        interaction = DecayingDropout(initial_keep_rate=dropout_initial_keep_rate,
                                      decay_interval=dropout_decay_interval,
                                      decay_rate=dropout_decay_rate)(interaction)
        '''Feature Extraction layer'''
        feature_extractor_input = Conv2D(filters=int(d * FSDR), kernel_size=1, activation=None, name='FSD')(interaction)
        feature_extractor = DenseNet(include_top=False,
                                     input_tensor=Input(shape=K.int_shape(feature_extractor_input)[1:]),
                                     nb_dense_block=nb_dense_blocks,
                                     nb_layers_per_block=n,
                                     compression=TSDR,
                                     growth_rate=GR)(feature_extractor_input)

        '''Output layer'''
        features = DecayingDropout(initial_keep_rate=dropout_initial_keep_rate,
                                   decay_interval=dropout_decay_interval,
                                   decay_rate=dropout_decay_rate,
                                   name='Features')(feature_extractor)
        out = Dense(units=nb_labels, activation='softmax', name='Output')(features)
        super(DIIN, self).__init__(inputs=[premise_word_input,    premise_char_input,    premise_syntactical_input,
                                           hypothesis_word_input, hypothesis_char_input, hypothesis_syntactical_input],
                                   outputs=out,
                                   name=name)
