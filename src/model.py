import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os.path as osp

import utils
import os

from tensorlayer.layers import retrieve_seq_length_op2, DenseLayer, Seq2Seq, EmbeddingInputlayer
from time import time
from sklearn.utils import shuffle


class Model:
    def __init__(self, config, *, train=False, reuse=True):
        # copy create_model stuff to here, and some other stuff
        self.sess = None
        self.out = None
        self.rnn = None
        self.config = config
        self.train = train
        self.reuse = reuse

    def load_data(self, path, silent=False):
        """
        """
        data_mod = utils.load_data_module(path)
        metadata, idx_q, idx_a = data_mod.load_data(path)
        data = list(data_mod.split_dataset(idx_q, idx_a))

        for i, (x, y) in enumerate(data):
            data[i] = (tl.prepro.remove_pad_sequences(x.tolist()), tl.prepro.remove_pad_sequences(y.tolist()))

        (self.train_x, self.train_y) = data[0]
        (self.test_x, self.test_y) = data[1]
        (self.valid_x, self.valid_y) = data[2]

        xseq_len = len(self.train_x)
        yseq_len = len(self.train_y)

        assert xseq_len == yseq_len

        self.n_step = int(xseq_len / self.config.batch_size)
        self.word2index = metadata['word2index']
        self.index2word = metadata['index2word']
        self.unk_id = self.word2index['unk']
        self.pad_id = self.word2index['_']

        self.xvocab_size = len(metadata['index2word'])
        self.start_id = self.xvocab_size
        self.end_id = self.xvocab_size + 1

        self.word2index.update({'start_id': self.start_id, 'end_id': self.end_id})

        self.index2word += ['start_id', 'end_id']
        self.xvocab_size += 2

        if not silent:
            print(f'encode_seqs: {[self.index2word[i] for i in self.train_x[10]]}')

            target_seqs = tl.prepro.sequences_add_end_id([self.train_y[10]], end_id=self.end_id)[0]
            print(f'target_seqs: {[self.index2word[i] for i in target_seqs]}')

            decode_seqs = tl.prepro.sequences_add_start_id([self.train_y[10]], start_id=self.start_id, remove_last=False)[0]
            print(f'decode_seqs: {[self.index2word[i] for i in decode_seqs]}')

            target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]
            print(f'target_mask: {target_mask}')

            print(len(target_seqs), len(decode_seqs), len(target_mask))

        batch_size = self.config.batch_size

        self.encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='encode_seqs')
        self.decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='decode_seqs')
        self.target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='target_seqs')
        self.target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='target_mask')

        if not self.train:
            self.encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name='encode_seqs')
            self.decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name='decode_seqs')
        else:
            self.encode_seqs2 = None
            self.decode_seqs2 = None

    def init_models(self):
        # this should be run after data is loaded and preprocessed
        with tf.variable_scope('model', reuse=self.reuse):
            with tf.variable_scope('embedding') as vs:
                net_encode = EmbeddingInputlayer(inputs=self.encode_seqs, vocabulary_size=self.xvocab_size,
                                                 embedding_size=self.config.embedding_size, name='seq_embedding')

                vs.reuse_variables()
                tl.layers.set_name_reuse(True)

                net_decode = EmbeddingInputlayer(inputs=self.decode_seqs, vocabulary_size=self.xvocab_size,
                                                 embedding_size=self.config.embedding_size, name='seq_embedding')

            net_rnn = Seq2Seq(net_encode, net_decode, cell_fn=tf.contrib.rnn.BasicLSTMCell,
                              n_hidden=self.config.embedding_size, initializer=tf.random_uniform_initializer(-0.1, 0.1),
                              encode_sequence_length=retrieve_seq_length_op2(self.encode_seqs),
                              decode_sequence_length=retrieve_seq_length_op2(self.decode_seqs),
                              initial_state_encode=None, dropout=0.5 if self.train else None, n_layer=3,
                              return_seq_2d=True, name='seq2seq')
            net_out = DenseLayer(net_rnn, n_units=self.xvocab_size, act=tf.identity, name='output')

        self.out = net_out
        self.rnn = net_rnn
        self.y_softmax = tf.nn.softmax(self.out.outputs)

        if self.train:
            self.loss = tl.cost.cross_entropy_seq_with_mask(logits=self.out.outputs, target_seqs=self.target_seqs,
                                                            input_mask=self.target_mask, return_details=False,
                                                            name='cost')
            self.out.print_params(False)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=sess_config)

        tl.layers.initialize_global_variables(self.sess)

    def load_state(self, silent=False):
        load_data = None

        if not osp.exists(self.config.output_path):
            os.makedirs(self.config.output_path)

        for name, _ in utils.get_latest_trained_data(self.config.output_path):
            try:
                load_data = np.load(name)['params']

                tl.files.assign_params(self.sess, load_data, self.out)

                if not silent:
                    print(f'Successfully loaded checkpoint file "{name}"')

                break
            except IOError:
                continue

        if load_data is None and not silent:
            print('Failed to load any trained checkpoint files. Starting afresh.')

    def save_state(self, epoch=None):
        """
        Saves the current state of the model and params.
        """
        if epoch is None:
            tl.files.save_npz(self.out.all_params, f'{self.config.output_path}/{self.config.name_template}.npz',
                              sess=self.sess)
        else:
            if epoch % self.config.checkpoint_step == 0:
                tl.files.save_npz(self.out.all_params, f'{self.config.output_path}/{self.config.name_template}'
                                  f'{epoch}.npz', sess=self.sess)
            else:
                tl.files.save_npz(self.out.all_params, f'{self.config.output_path}/{self.config.name_template}.npz',
                                  sess=self.sess)

    def train_epoch(self, epoch, inference_model):
        if not self.train:
            raise Exception('Model is not set up for training.')

        epoch_time = time()

        # Shuffle training data.
        print(self.train)
        self.train_x, self.train_y = shuffle(self.train_x, self.train_y, random_state=0)
        total_err = 0
        n_iter = 0

        for x, y in tl.iterate.minibatches(inputs=self.train_x, targets=self.train_y, batch_size=self.config.batch_size,
                                           shuffle=False):
            step_time = time()
            x = tl.prepro.pad_sequences(x)
            _target_seqs = tl.prepro.sequences_add_end_id(y, end_id=self.end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)
            _decode_seqs = tl.prepro.sequences_add_start_id(y, start_id=self.start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            print(self.target_mask)
            print(_target_mask)

            _, err = self.sess.run([self.train_op], {
                self.encode_seqs: x,
                self.decode_seqs: _decode_seqs,
                self.target_seqs: _target_seqs,
                self.target_mask: _target_mask
            })

            print(f'Epoch[{epoch/self.config.epochs}] step[{n_iter}/{self.n_step}] loss:{err}'
                  f' took: {time() - step_time:.5}s')

            total_err += err
            n_iter += 1

            if n_iter and n_iter % 1000 == 0 or n_iter == self.n_step:
                for seed in self.config.seeds:
                    print(f'Query > "{seed}"')

                    for _ in range(5):  # 5 replies for each seed.
                        # Encode and get state.
                        sentence = inference_model.talk(seed, self.sess)

                        print(f' > "{sentence}"')

        print(f'Epoch[{epoch}/{self.config.epochs}] averaged loss: {total_err / n_iter}'
              f'took: {time() - epoch_time:.5}s')

    def talk(self, input, sess=None):
        if self.train:
            raise Exception('Cannot talk while in training mode.')

        sess = sess or self.sess
        seed_id = [self.word2index[w] for w in input.split(' ')]

        state = sess.run(self.rnn.final_state_encode, {
            self.encode_seqs2: [seed_id]
        })

        o, state = sess.run([self.y_softmax, self.rnn.final_state_decode], {
            self.rnn.initial_state_decode: state,
            self.decode_seqs2: [[self.start_id]]
        })
        w_id = tl.nlp.sample_top(o[0], top_k=3)
        w = self.index2word[w_id]
        sentence = [w]

        for _ in range(self.config.reply_length):
            o, state = sess.run([self.y_softmax, self.rnn.final_state_decode], {
                self.rnn.initial_state_decode: state,
                self.decode_seqs2: [[w_id]]
            })
            w_id = tl.nlp.sample_top(o[0], top_k=2)

            if w_id == self.end_id:
                break

            w = self.index2word[w_id]
            sentence += [w]

        return ' '.join(sentence)
