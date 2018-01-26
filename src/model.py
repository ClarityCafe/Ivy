import tensorflow as tf

from textdata import Batch


class ProjectionOp:
    """
    Single layer perceptron.
    Project input tensor on the output dimension.
    """

    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape (tuple): (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        if len(shape) != 2:
            raise ValueError(f'shapes must have a size of 2. Got size {len(shape)}')

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + scope):
            self.wt = tf.get_variable('weights', shape, dtype=dtype)
            self.b = tf.get_variable('bias', shape[0], initializer=tf.constant_initializer(), dtype=dtype)
            self.w = tf.transpose(self.wt)

    def get_weights(self):
        """
        Convenience method for some tf arguments.
        """
        return self.w, self.b

    def __call__(self, x):
        """
        Project the output of the decoder into the vocabular space.
        Args:
            x (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(x, self.w) + self.b


class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, config, text_data):
        print('Model creation...')

        self.text_data = text_data  # Keep a reference on the dataset.
        self.config = config  # Keep track of the parameters of the model.
        self.dtype = tf.float32

        # Placeholdeers
        self.encoder_inputs = None
        self.decoder_inputs = None  # same that decoder_target plus the <go>.
        self.decoder_targets = None
        self.decoder_weights = None  # Adjust the learning to the target sentence size.

        # Main operators
        self.loss_fct = None
        self.opt_op = None
        self.ouputs = None  # outputs of the network, list of probability for each word.

        # Construct graphs.
        self.build_network()

    def build_network(self):
        """
        Creates the computational graph.
        """

        # Parameters of sample softmax (needed for attention mechanism and a large vocabulary size).
        output_projection = None
        vocab_size = self.text_data.get_vocab_size()

        # Sampled softmax only makes sense if we sample less than the vocabulary size.
        if 0 < self.config.network.softmax_samples < vocab_size:
            output_projection = ProjectionOp((vocab_size, self.config.network.hidden_size), 'softmax_projection', self.dtype)

            def sampled_softmax(labels, logits):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1).

                # We need to compute the sampled_softmax_loss using 32bit floats to avoid numerical instabilities
                local_wt = tf.cast(output_projection.wt, tf.float32)
                local_b = tf.cast(output_projection.b, tf.float32)
                local_logits = tf.cast(logits, tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(local_wt, local_b, labels, local_logits, self.config.network.softmax_samples, vocab_size),
                    tf.int32)

        # Creation of the rnn cell
        def create_rnn_cell():
            enco_deco_cell = tf.contrib.rnn.BasicLSTMCell(self.config.network.hidden_size)

            if self.config.general.train:  # TODO: Should use a placeholder instead.
                enco_deco_cell = tf.contrib.rnn.DropoutWrapper(enco_deco_cell, input_keep_prob=1.0, output_keep_prob=self.config.training.dropout)

            return enco_deco_cell

        enco_deco_cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(self.config.network.num_layers)])

        # Network input (placeholders).

        with tf.name_scope('placeholder_encoder'):
            self.encoder_inputs = [tf.placeholder(tf.int32, [None]) for _ in range(self.config.dataset.max_length_enco)]

        with tf.name_scope('placeholder_decoder'):
            self.decoder_inputs = [tf.placeholder(tf.int32, [None], name='inputs') for _ in range(self.config.dataset.max_length_deco)]
            self.decoder_targets = [tf.placeholder(tf.int32, [None], name='targets') for _ in range(self.config.dataset.max_length_deco)]
            self.decoder_weights = [tf.placeholder(tf.float32, [None], name='weights') for _ in range(self.config.dataset.max_length_deco)]

        # Define the network.
        # Here we use an embedding model, it takes integers as inputs, and converts them into word vectors for better word representation.
        decoder_outputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoder_inputs,  # List<[batch=?, input_dim=1]>, list of size args.max_length.
            self.decoder_inputs,  # For training, we force the correct output (feed_previous=False).
            enco_deco_cell,
            vocab_size,
            vocab_size,  # Both encoder and decoder have the same number of classes.
            embedding_size = self.config.network.embedding_size,  # Dimension of each word.
            output_projection=output_projection.get_weights() if output_projection else None,
            feed_previous=not self.config.general.train  # When we test (not self.config.general.train), we use previous output as next input (feed_previous)
        )

        # TODO: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046): should speed up training and reduce memory usage.
        # Other solution, use sampling softmax.

        if not self.config.general.train:  # For testing only
            if not output_projection:
                self.outputs = decoder_outputs
            else:
                self.outputs = [output_projection(output) for output in decoder_outputs]
        else:  # For training only
            # Finally, we define the loss function.
            self.loss_fct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoder_outputs,
                self.decoder_targets,
                self.decoder_weights,
                vocab_size,
                softmax_loss_function=sampled_softmax if output_projection else None  # If None, use default softmax.
            )

            # Keep track of the cost.
            tf.summary.scalar('loss', self.loss_fct)

            # Initalise the optimiser.
            opt = tf.train.AdamOptimizer(learning_rate=self.config.training.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
            self.opt_op = opt.minimize(self.loss_fct)

    def step(self, batch):
        """
        Forward/training step operation.
        Does not perform run on itself but just return the operators to do so.
        Those then have to be run.

        Args:
            batch (Batch): Input data in testing mode, input and target in output mode.
        Returns:
            (ops), dict: A tuple of the (training, loss) operators, or (outputs,) in testing mode with the associated feed dictionary.
        """

        # Feed ~~me~~ the dictionary.
        feed_dict = {}
        ops = None

        if self.config.general.train:  # Training
            for i in range(self.config.dataset.max_length_enco):
                feed_dict[self.encoder_inputs[i]] = batch.encoder_seqs[i]

            for i in range(self.config.dataset.max_length_deco):
                feed_dict[self.decoder_inputs[i]] = batch.decoder_seqs[i]
                feed_dict[self.decoder_targets[i]] = batch.target_seqs[i]
                feed_dict[self.decoder_weights[i]] = batch.weights[i]
        else:  # Testing (batch_size == 1)
            for i in range(self.config.dataset.max_length_enco):
                feed_dict[self.encoder_inputs[i]] = batch.encoder_seqs[i]

            feed_dict[self.decoder_inputs[0]] = [self.text_data.go_token]
            ops = (self.outputs,)

        # Return one pass operator
        return ops, feed_dict