import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os.path as osp

import model
import utils
import yaml
import time
import os

from sklearn.utils import shuffle

with open('config.yaml') as f:
    config = utils.AttrObj(yaml.load(f))

data_module = utils.load_data_module(config.training.data_path)
metadata, idx_q, idx_a = data_module.load_data(config.training.data_path)
data = list(data_module.split_dataset(idx_q, idx_a))

for i, (x, y) in enumerate(data):
    data[i] = (tl.prepro.remove_pad_sequences(x.tolist()), tl.prepro.remove_pad_sequences(y.tolist()))

(train_x, train_y), (test_x, test_y), (valid_x, valid_y) = data

xseq_len = len(train_x)
yseq_len = len(train_y)

assert xseq_len == yseq_len

n_step = int(xseq_len / config.training.batch_size)
xvocab_size = len(metadata['index2word'])

word2index = metadata['word2index']
index2word = metadata['index2word']
unk_id = word2index['unk']
pad_id = word2index['_']

start_id = xvocab_size
end_id = xvocab_size + 1

word2index.update({'start_id': start_id, 'end_id': end_id})

index2word += ['start_id', 'end_id']
xvocab_size = yvocab_size = xvocab_size + 2

""" A data for Seq2Seq should look like this:
input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
target_mask : [1, 1, 1, 1, 0]
"""

print('encode_seqs', [index2word[i] for i in train_x[10]])

target_seqs = tl.prepro.sequences_add_end_id([train_y[10]], end_id=end_id)[0]
print('target_seqs', [index2word[i] for i in target_seqs])

decode_seqs = tl.prepro.sequences_add_start_id([train_y[10]], start_id=start_id, remove_last=False)[0]
print('decode_seqs', [index2word[i] for i in decode_seqs])

target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]
print('target_mask', target_mask)

print(len(target_seqs), len(decode_seqs), len(target_mask))

# Models for training
encode_seqs = tf.placeholder(dtype=tf.int64, shape=[config.training.batch_size, None], name='encode_seqs')
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[config.training.batch_size, None], name='decode_seqs')
target_seqs = tf.placeholder(dtype=tf.int64, shape=[config.training.batch_size, None], name='target_seqs')
target_mask = tf.placeholder(dtype=tf.int64, shape=[config.training.batch_size, None], name='target_mask')
train_net, _ = model.create_model(encode_seqs, decode_seqs, xvocab_size)

# Models for inferencing
encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name='encode_seqs')
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name='decode_seqs')
net, net_rnn = model.create_model(encode_seqs2, decode_seqs2, xvocab_size, train=False, reuse=True)
y_ = tf.nn.softmax(net.outputs)

# Loss for training
loss = tl.cost.cross_entropy_seq_with_mask(logits=train_net.outputs, target_seqs=target_seqs, input_mask=target_mask,
                                           return_details=False, name='cost')

train_net.print_params(False)

train_op = tf.train.AdamOptimizer(learning_rate=config.training.learning_rate).minimize(loss)
sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=sess_config)

tl.layers.initialize_global_variables(sess)

load_data = None

if not osp.exists(config.training.output_path):
    os.makedirs(config.training.output_path)

for name, _ in utils.get_latest_trained_data(config.training.output_path):
    try:
        load_data = np.load(name)['params']

        tl.files.assign_params(sess, load_data, net)
        print(f'Successfully loaded checkpoint file "{name}"')

        break
    except IOError:
        continue

if load_data is None:
    print('Failed to load any trained checkpoint files. Starting fresh.')

for epoch in range(config.training.epochs):
    epoch_time = time.time()

    # Shuffle training data.
    train_x, train_y = shuffle(train_x, train_y, random_state=0)
    total_err = 0
    n_iter = 0

    for x, y in tl.iterate.minibatches(inputs=train_x, targets=train_y, batch_size=config.training.batch_size,
                                       shuffle=False):
        step_time = time.time()
        x = tl.prepro.pad_sequences(x)
        _target_seqs = tl.prepro.sequences_add_end_id(y, end_id=end_id)
        _target_seqs = tl.prepro.pad_sequences(_target_seqs)
        _decode_seqs = tl.prepro.sequences_add_start_id(y, start_id=start_id, remove_last=False)
        _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
        _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

        _, err = sess.run([train_op, loss], {
            encode_seqs: x,
            decode_seqs: _decode_seqs,
            target_seqs: _target_seqs,
            target_mask: _target_mask
        })

        print(f'Epoch[{epoch}/{config.training.epochs}] step[{n_iter}/{n_step}] loss:{err}'
              f' took:{time.time() - step_time:.5}s')

        total_err += err
        n_iter += 1

        # Test every 1000 steps
        if n_iter and n_iter % 1000 == 0 or n_iter == n_step:
            for seed in config.training.seeds:
                print(f'Query > "{seed}"')

                seed_id = [word2index[w] for w in seed.split(' ')]

                for _ in range(5):  # 1 query => 5 replies
                    # Encode and get state.
                    state = sess.run(net_rnn.final_state_encode, {
                        encode_seqs2: [seed_id]
                    })

                    # Decode, feed start_id and get first word.
                    # noqa Reference: https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
                    o, state = sess.run([y_, net_rnn.final_state_decode], {
                        net_rnn.initial_state_decode: state,
                        decode_seqs2: [[start_id]]
                    })
                    w_id = tl.nlp.sample_top(o[0], top_k=3)
                    w = index2word[w_id]

                    # Decode, feed state iteratively.
                    sentence = [w]

                    for _ in range(config.training.reply_length):
                        o, state = sess.run([y_, net_rnn.final_state_decode], {
                            net_rnn.initial_state_decode: state,
                            decode_seqs2: [[w_id]]
                        })
                        w_id = tl.nlp.sample_top(o[0], top_k=2)

                        if w_id == end_id:
                            break

                        w = index2word[w_id]
                        sentence += [w]

                    print(f" > \"{' '.join(sentence)}\"")

    print(f'Epoch[{epoch}/{config.training.epochs}] averaged loss:{total_err / n_iter}'
          f'took:{time.time() - epoch_time:.5}s')

    if epoch % config.training.checkpoint_step == 0:
        tl.files.save_npz(net.all_params, f'{config.training.output_path}/{config.training.name_template}{epoch}.npz',
                          sess=sess)
    else:
        tl.files.save_npz(net.all_params, f'{config.training.output_path}/{config.training.name_template}.npz',
                          sess=sess)
