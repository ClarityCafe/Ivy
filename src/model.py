import tensorflow as tf
import tensorlayer as tl

import yaml
import utils

from tensorlayer.layers import retrieve_seq_length_op2, DenseLayer, Seq2Seq, EmbeddingInputlayer

with open('config.yaml') as f:
    config = utils.AttrObj(yaml.load(f))


def create_model(encode_seqs, decode_seqs, xvocab_size, *, train=True, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        with tf.variable_scope('embedding') as vs:
            net_encode = EmbeddingInputlayer(inputs=encode_seqs, vocabulary_size=xvocab_size,
                                             embedding_size=config.training.embedding_size, name='seq_embedding')

            vs.reuse_variables()
            tl.layers.set_name_reuse(True)

            net_decode = EmbeddingInputlayer(inputs=decode_seqs, vocabulary_size=xvocab_size,
                                             embedding_size=config.training.embedding_size, name='seq_embedding')

        net_rnn = Seq2Seq(net_encode, net_decode, cell_fn=tf.contrib.rnn.BasicLSTMCell,
                          n_hidden=config.training.embedding_size, initializer=tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length=retrieve_seq_length_op2(encode_seqs),
                          decode_sequence_length=retrieve_seq_length_op2(decode_seqs), initial_state_encode=None,
                          dropout=0.5 if train else None, n_layer=3, return_seq_2d=True, name='seq2seq')
        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')

    return net_out, net_rnn
