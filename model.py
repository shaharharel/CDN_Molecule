import tensorflow as tf


class CDN:

    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters, max_molecule_length,
                 gaussian_samples, variational=True, l2_reg_lambda=0.5, generation_mode=False, test_mode=False):

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.max_molecule_length = max_molecule_length
        self.l2_reg_lambda = l2_reg_lambda
        self.gaussian_samples_dim = gaussian_samples
        self.variational = variational
        self.encoder_input_GO = tf.placeholder(tf.int32, [None, sequence_length], name="encoder_input")
        self.encoder_input = tf.placeholder(tf.int32, [None, sequence_length], name="encoder_input")  ## no go
        self.gaussian_samples = tf.placeholder(tf.float32, [None, self.gaussian_samples_dim], name="unit_gaussians")
        self.generation_mode = generation_mode
        self.test_mode = test_mode

    def encode(self):
        # Embedding layer
        with tf.name_scope("embedding"):
            self.E = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.E, self.encoder_input)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_go = tf.nn.embedding_lookup(self.E, self.encoder_input_GO)

        # Create a convolution layers for each filter size
        conv_flatten = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                conv_flatten.append(tf.contrib.layers.flatten(h))
        conv_output = tf.concat(conv_flatten, axis=1)

        # Flatten feature vector
        h_pool_flat3 = tf.nn.relu(tf.contrib.layers.linear(conv_output, 450))

        if self.variational:
            with tf.name_scope("Variational"):
                self.z_mean = tf.contrib.layers.linear(h_pool_flat3, 300)
                self.z_stddev = tf.contrib.layers.linear(h_pool_flat3, 300)
                latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) -
                                                       tf.log(tf.square(self.z_stddev)) - 1, 1)
                self.mean_latent_loss = tf.reduce_mean(latent_loss)
                if self.generation_mode:
                    h_pool_flat = self.gaussian_samples
                else:
                    h_pool_flat = self.z_mean + (self.z_stddev * self.gaussian_samples)

                h_pool_flat = tf.identity(h_pool_flat, "encoded_final")

        return h_pool_flat, self.mean_latent_loss

    def decode_rnn(self, z):
        def pick_next_argmax(former_output, step):
            next_symbol = tf.expand_dims(tf.stop_gradient(tf.argmax(former_output, 1)), axis=-1)
            return tf.nn.embedding_lookup(self.E, next_symbol), next_symbol

        def pick_next_top_k(former_output, step):
            next_symbol = tf.multinomial(former_output, 1)
            return tf.nn.embedding_lookup(self.E, next_symbol), next_symbol

        with tf.name_scope("Decoder"):
            self.decode_start = tf.nn.relu(tf.contrib.layers.linear(z, 150))
            decoder_inputs_list = tf.split(self.embedded_chars_go, self.max_molecule_length, axis=1)
            decoder_inputs_list = [tf.squeeze(i, axis=1) for i in decoder_inputs_list]
            rnn_cell = tf.nn.rnn_cell.LSTMCell(150, state_is_tuple=False)

            self.lstm_outputs = []
            temp_logits = []
            self.all_symbols = []
            symbol = tf.ones(1)  # output for test mode
            for i in range(self.max_molecule_length):
                if not self.test_mode or i == 0:
                    if i == 0:
                        output, state = rnn_cell(decoder_inputs_list[i], state=z)
                    else:
                        output, state = rnn_cell(decoder_inputs_list[i], state=state)
                else:
                    next_decoder_input, symbol = pick_next_argmax(temp_logits[-1], i)
                    next_decoder_input = tf.squeeze(next_decoder_input, axis=1)
                    output, state = rnn_cell(next_decoder_input, state=state)
                with tf.variable_scope("decoder_output_to_logits") as scope_logits:
                    if i > 0:
                        scope_logits.reuse_variables()
                    temp_logits.append(tf.contrib.layers.linear(output, self.vocab_size))

                self.lstm_outputs.append(output)
                if i > 0:
                    self.all_symbols.append(symbol)
                if i == self.max_molecule_length - 1 and self.test_mode:
                    self.all_symbols.append(pick_next_argmax(temp_logits[-1], i)[1])
            if self.test_mode:
                self.all_symbols = tf.squeeze(tf.transpose(tf.stack(self.all_symbols), [1,0,2]), axis=-1)

            self.decoder_logits = tf.transpose(tf.stack(temp_logits), perm=[1, 0, 2])
            self.decoder_prediction = tf.argmax(self.decoder_logits, 2, name="decoder_predictions")

            return self.decoder_logits

    def loss(self, logits, latent_loss):
        with tf.name_scope("loss"):
            self.output_onehot = tf.one_hot(self.encoder_input, self.vocab_size)
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.output_onehot)
            self.CE_loss = tf.reduce_mean(self.losses)
            self.total_loss = self.CE_loss + .00001 * latent_loss

        with tf.name_scope("accuracy"):
            decoder_prediction = tf.argmax(logits, 2, name="decoder_predictions")
            x_target = tf.to_int64(self.encoder_input)
            correct_predictions = tf.equal(decoder_prediction, x_target)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        return self.total_loss, self.accuracy