import numpy as np

import random
import nltk
import itertools
import pickle

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
FILENAME = 'monika-script.txt'
UNK = 'unk'
VOCAB_SIZE = 6000
LIMIT = {
    'maxq': 25,
    'minq': 2,
    'maxa': 25,
    'mina': 2
}


def read_lines(filename):
    with open(filename, 'rb') as f:
        return f.read().decode('utf8').split('\n')[:-1]


def split_line(line):
    return line.split('.')


def filter_line(line, whitelist):
    return ''.join(c for c in line if c in whitelist)


def create_index(tokenized, vocab_size):
    # Get frequency distribution.
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized))
    # Get vocabulary of 'vocab_size' most used words.
    vocab = freq_dist.most_common(vocab_size)
    index2word = ['_', UNK] + [x[0] for x in vocab]
    word2index = dict((w, i) for i, w in enumerate(index2word))

    return index2word, word2index, freq_dist


def filter_data(seqs):
    filtered_q = []
    filtered_a = []
    raw_data_len = len(seqs) // 2

    for i in range(0, len(seqs), 2):
        qlen = len(seqs[i].split(' '))
        alen = len(seqs[i - 1].split(' '))

        if (qlen >= LIMIT['minq'] and qlen <= LIMIT['maxq']) and (alen >= LIMIT['mina'] and alen <= LIMIT['maxa']):
            filtered_q.append(seqs[i])
            filtered_a.append(seqs[i - 1])

    # Print the fraction of the original data, filtered.
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(f'{filtered}% filtered from original data')

    return filtered_q, filtered_a


def zero_pad(q_tokens, a_tokens, word2index):
    # Number of rows.
    data_len = len(q_tokens)

    # Numpy arrays to store indices
    idx_q = np.zeros([data_len, LIMIT['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, LIMIT['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(q_tokens[i], word2index, LIMIT['maxq'])
        a_indices = pad_seq(a_tokens[i], word2index, LIMIT['maxa'])

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


def pad_seq(seq, lookup, max_len):
    indices = []

    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])

    return indices + [0] * (max_len - len(seq))


def process_data():
    print('\n>> Read lines from file')

    lines = read_lines(FILENAME)
    lines = [line.lower() for line in lines]  # Change to lowercase.

    print('\n:: Sample from read(p) lines')
    print(lines[121:125])

    # Filter out unnecessary characters
    print('\n>> Filter lines')
    lines = [filter_line(line, EN_WHITELIST) for line in lines]

    print(lines[121:125])

    # Filter out too long or too short sequences.
    print('\n>> 2nd layer of filtering')

    q_lines, a_lines = filter_data(lines)

    print(f'\nq : {q_lines[60]} ; a : {a_lines[60]}')
    print(f'\nq : {q_lines[61]} ; a : {a_lines[61]}')

    # Convert list of [lines of text] into list of [list of words]
    print('\n>> Segment lines into words')

    q_tokens = [x.split(' ') for x in q_lines]
    a_tokens = [x.split(' ') for x in a_lines]

    print('\n:: Sample from segmented list of words')
    print(f'\nq : {q_tokens[60]} ; a {a_tokens[60]}')
    print(f'\nq : {q_tokens[61]} ; a {a_tokens[61]}')

    # Indexing -> index2word, word2index, en/ta
    print('\n >> Index words')
    index2word, word2index, freq_dist = create_index(q_tokens + a_tokens, VOCAB_SIZE)

    print('\n >> Zero padding')
    idx_q, idx_a = zero_pad(q_tokens, a_tokens, word2index)

    print('\n >> Save numpy arrays to disk')
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # Save various dicts for later.
    metadata = {
        'word2index': word2index,
        'index2word': index2word,
        'limit': LIMIT,
        'freq_dist': freq_dist
    }

    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def load_data(path=''):
    # Read data control dicts.
    try:
        with open(path + '/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except Exception:
        metadata = None

    # Read numpy arrays
    idx_q = np.load(path + '/idx_q.npy')
    idx_a = np.load(path + '/idx_a.npy')

    return metadata, idx_q, idx_a


def split_dataset(x, y, ratio=[0.7, 0.15, 0.15]):
    # Number of examples.
    data_len = len(x)
    lens = [int(data_len * item) for item in ratio]

    train_x = x[:lens[0]]
    train_y = y[:lens[0]]
    test_x = x[lens[0]:lens[0] + lens[1]]
    test_y = y[lens[0]:lens[0] + lens[1]]
    valid_x = x[-lens[-1]:]
    valid_y = y[-lens[-1]:]

    return (train_x, train_y), (test_x, test_y), (valid_x, valid_y)


def ran_batch_gen(x, y, batch_size):
    while True:
        sample_idx = random.sample(list(np.arrange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T


def decode(seq, lookup, separator=''):  # 0 isused for padding, is ignored
    return separator.join(lookup[e] for e in seq if e)


if __name__ == '__main__':
    process_data()
