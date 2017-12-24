import os.path as osp
import numpy as np

import os
import collections
import pickle

from bz2 import BZ2File

class TextLoader:
    '''Class used for loading text from a file.'''

    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.tensor_sizes = []
        self.tensor_file_template = osp.join(data_dir, 'data{}.npz')
        self.input_files = self._get_input_file_list(data_dir)
        self.input_file_count = len(self.input_files)

        vocab_file = osp.join(data_dir, 'vocab.pkl')
        sizes_file = osp.join(data_dir, 'sizes.pkl')

        if self.input_file_count < 1:
            raise ValueError('Input files not found. File names must either end in ".txt" or ".bz2".')

        if self._preprocess_required(vocab_file, sizes_file, self.tensor_file_template, self.input_file_count):
            # If either the vocab file or the tensor file doesn't already exist, create them.
            print(f'Preprocessing the following files: {self.input_files}')

            vocab_counter = collections.Counter()

            for file_ in self.input_files:
                print(f'Reading vocab from input file {file_}')

                self._augment_vocab(vocab_counter, file_)

            print('Saving vocab file.')
            self._save_vocab(vocab_counter, vocab_file)

            for i, file_ in enumerate(self.input_files):
                print(f'Preprocessing input file {file_}')
                self._preprocess(file_, self.tensor_file_template.format(i))
                self.tensor_sizes.append(self.tensor.size)

            with open(sizes_file, 'wb') as f:
                pickle.dump(self.tensor_sizes, f)

            print(f'Processed input text file: {self.tensor.size} characters loaded.')
        else:
            # If the vocab file and sizes file already exist, load them.
            print('Loading vocab file.')
            self._load_vocab(vocab_file)
            print('Loading sizes file.')

            with open(sizes_file, 'rb') as f:
                self.tensor_sizes = pickle.load(f)

        self.tensor_batch_counts = [n / (self.batch_size * self.seq_length) for n in self.tensor_sizes]
        self.total_batch_count = sum(self.tensor_batch_counts)

        print(f'Total batch count: {self.total_batch_count}')

        self.tensor_index = -1

    def _preprocess_required(self, vocab_file, sizes_file, tensor_file_template, input_file_count):
        if not osp.exists(vocab_file):
            print('No vocab file found. Preprocessing...')
            return True
        elif not osp.exists(sizes_file):
            print('No sizes file found. Preprocessing...')
            return True

        for i in range(input_file_count):
            if not osp.exists(tensor_file_template.format(i)):
                print(f"Couldn't find {tensor_file_template.format(i)}. Preprocessing...")
                return True

        return False

    def _get_input_file_list(self, data_dir):
        suffixes = ['.txt', '.bz2']
        input_file_list = []

        if osp.isdir(data_dir):
            for root, dir_, files in os.walk(data_dir):
                for filename in files:
                    if filename.startswith('.'):
                        continue

                    filepath = osp.join(root, filename)

                    if filepath.endswith(suffixes[0]) or filepath.endswith(suffixes[1]):
                        input_file_list.append(filepath)
        else:
            raise ValueError(f'{data_dir} is not a directory.')

        return sorted(input_file_list)

    def _augment_vocab(self, vocab_counter, input_file):
        # Load up the input.txt file and use it to create a vocab file and a tensor file at the specified file paths.
        if input_file.endswith('.bz2'):
            file_ref = BZ2File
        elif input_file.endswith('.txt'):
            file_ref = open

        with file_ref(input_file, 'r') as f:
            data = f.read()
            data = data.decode(self.encoding)

            vocab_counter.update(data)

    def _save_vocab(self, vocab_counter, vocab_file):
        # count_pairs is a list of these dictionary entries, sorted in descending order.
        # The first item of the list is a 2-item tuple of the most common character and the number of times it occurs, then the second-most common, etc. -- e.g.:
        # [(' ', 17), ('a', 11), ('e', 7), ('n', 7), ...]
        count_pairs = sorted(vocab_counter.items(), key=lambda x: -x[1])

        # self.chars is a tuple (immutable ordered list) of characters, in descending order from most common to least. E.g.:
        # (' ', 'a', 'e', 'n', 't', ...)
        # This is a lookup device to convert index number to character.
        # How does this work?
        # zip(*___) returns an iterator of tuples, where the i-th tuple contains the i-th element from each of the argument sequences or iterables.
        # So zip(*count_pairs) returns an iterator over two tuples, the first tuple being characters in descending order of frequency, and the second being the frequency of the same characters.
        # list() then packages these two tuples into a list of the same two tuples, and the assignment passes the first tuple (characters in descending order) to self.chars and the second (character counts) to a disregarded variable.
        self.chars, _ = list(zip(*count_pairs))

        # self.vocab_size counts the number of characters used in input.txt.
        self.vocab_size = len(self.chars)

        # self.vocab is a dictionary that maps each character to its index number.
        # For example:
        # [(' ', 0), ('a', 1), ('e', 2), ('n', 3), ...]
        # This is a lookup device to convert a character to its index number.
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        # Save the characters tuple to vocab.pkl (tiny file).
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)

        print(f'Saved vocab of size {self.vocab_size}')

    def _load_vocab(self, vocab_file):
        # Load the character tuple (vocab.pkl) to self.chars.
        # Remember that it is in descending order of character frequency in the data.
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)

        # Use the character tuple to regenerate vocab_size and the vocab dictionary.
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def _preprocess(self, input_file, tensor_file):
        if input_file.endswith('.bz2'):
            file_ref = BZ2File
        elif input_file.endswith('.txt'):
            file_ref = open

        with file_ref(input_file, 'r') as f:
            data = f.read()
            data = data.decode(self.encoding)

        # Convert the entirety of the data file from charactgers to indices via the vocab dictionary.
        # How? map(function, iterable) returns a list of the output of the function executed on each member of the iterable.
        # E.g.:
        # [14, 2, 9, 2, 0, 6, 7, 0, ...]
        self.tensor = np.array([self.vocab.get(x) for x in data])

        # Compress and save the numpy tensor array to data.npz.
        np.savez_compressed(tensor_file, tensor_data=self.tensor)

    def _load_preprocessed(self, tensor_index):
        self.reset_batch_pointer()

        if tensor_index == self.tensor_index:
            return

        print(f'Loading tensor data file "{tensor_index}"')

        tensor_file = self.tensor_file_template.format(tensor_index)

        # Load the data tensor file to self.tensor.
        with np.load(tensor_file) as f:
            self.tensor = f['tensor_data']

        self.tensor_index = tensor_index

        # Calculate the number of batches in the data.
        # Each batch is batch_size x seq_length, so this is just the input data size divided by that product, rounded down.
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)

        if self.tensor_batch_counts[tensor_index] != self.num_batches:
            print(f'Error in batch size! Expected {self.tensor_batch_counts[tensor_index]}; found {self.num_batches}')

        # Chop off the end of the data tensor so that the length of the data is a whole multiple of the (batch_size x seq_length) product.
        # Do this with the slice operator on the numpy array.
        self.tensor = self.tensor[:int(self.num_batches * self.batch_size * self.seq_length)]

        # Construct two numpy arrays to represent input characters (xdata) and target characters (ydata).
        # In training, we will feed in input characters one at a time, and optimize along a loss function computed against the target characters.
        # (We do this with batch_size characters at a time, in parallel.)
        # Since this is a sequence prediction net, the target is just the input right-shifted by 1.
        xdata = self.tensor
        ydata = np.copy(self.tensor) # ydata starts as a copy of xdata
        ydata[:-1] = xdata[1:] # Right-shift y-data by 1 using the numpy array slice syntax.
        ydata[-1] = xdata[0] # Replace the very last character of y-data with the first character of the input data.

        # Split our unidemnsional data array into distinct batches.
        # How? xdata.reshape(self.batch_size, -1) returns a 2D numpy tensor view in which the first dimension is the batch index (from 0 to num_batches), and the second dimension is the index of the character within the batch (from 0 to (batch_size x seq_length)).
        # Within each batch, characters follow the same sequence as in the input data.
        # Then, np.split(that 2D numpy tensor, num_batches, 1) gives a list of numpy arrays.
        # Say batch_size = 4, seq_length = 5, and data is the following string: "Here is a new string named data. It is a new string named data. It is named data."
        # We truncate the string to lop off the last period (so there are now 80 characters, which is evenly divisible by 4 x 5). After xdata.reshape, we have:
        #
        # [[Here is a new string],
        #  [ named data. It is a],
        #  [ new string named da],
        #  [ta. It is named data]]
        #
        # After np.split, we have:
        # <[[Here ],   <[[is a ],   <[[new s],     <[[tring],
        #   [ name],     [d dat],     [a. It],       [ is a],
        #   [ new ],     [strin],     [g nam],       [ed da],
        #   [ta. I]]>,   [t is ]]>,   [named]]>,     [ data]]>
        #
        # where the first item of the list is the numpy array on the left.
        # Thus x_batches is a list of numpy arrays. The first dimension of each numpy array is the batch number (from 0 to batch_size), and the second dimension is the character index (from 0 to seq_length).
        #
        # These will be fed to the model one at a time sequentially.
        # State is preserved between sequential batches.
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        if self.tensor_index < 0:
            self._load_preprocessed(0)

        if self.pointer >= self.num_batches:
            self._load_preprocessed((self.tensor_index + 1) % self.input_file_count)

        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1

        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

    def cue_batch_pointer_to_epoch_fraction(self, epoch_fraction):
        self._cue_batch_pointer_to_step_count((epoch_fraction - int(epoch_fraction)) * self.total_batch_count)

    def _cue_batch_pointer_to_step_count(self, step_target):
        for i, n in enumerate(self.tensor_batch_counts):
            if step_target < n:
                break

            step_target -= n

        self.pointer = n
        self.current_tensor_index = i

        self._load_preprocessed(i)