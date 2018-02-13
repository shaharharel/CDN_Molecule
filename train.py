import tensorflow as tf
import numpy as np
import os
import time
import datetime
import model
import pickle
import preProcess

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .03, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "data/TrainVectors.pickle", "Data source.")
tf.flags.DEFINE_string("parameters_file", "runs/1518484761/checkpoints", "Checkpoint directory for training restart")

# Model Hyperparameters
tf.flags.DEFINE_integer("vocab_size", 37, "number of chars in SMILES vocab)")
tf.flags.DEFINE_integer("max_molecule_length", 50, "number of chars in SMILES vocab)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5,6", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda ")
tf.flags.DEFINE_float("unit_gaussian_dim", 300, "number of gaussians")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 50, "Number of checkpoints to store")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")
x = np.array(pickle.load(open(FLAGS.data_file, "rb")))

# Shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(x)))
x_shuffled = x[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(x)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(x_train), len(x_dev)))

load_param = False

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = model.CDN(
            sequence_length=x_train.shape[1],
            vocab_size=FLAGS.vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            max_molecule_length=FLAGS.max_molecule_length,
            gaussian_samples=FLAGS.unit_gaussian_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            variational=True,
            test_mode=False,
            generation_mode=False
        )

        global_step = tf.Variable(1, name="global_step", trainable=False)
        starter_learning_rate = 1e-3
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        encoded, latent_loss = cnn.encode()
        logits = cnn.decode_rnn(encoded)
        loss, accuracy = cnn.loss(logits, latent_loss)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss)
        acc_summary = tf.summary.scalar("accuracy", accuracy)
        kl_div_summary = tf.summary.scalar("KL-div", latent_loss)
        CE_loss_summary = tf.summary.scalar("CE-loss", cnn.CE_loss)
        learning_rate_summary = tf.summary.scalar("learningRate", learning_rate)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged, kl_div_summary, CE_loss_summary,
             learning_rate_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, kl_div_summary, CE_loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory.
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        if load_param:
            # model file to restart training
            checkpoint_file = 'runs/1516890001/checkpoints/model-64000'
            saver.restore(sess, checkpoint_file)
            print "Loaded: " + str(checkpoint_file)
        else:
            sess.run(tf.global_variables_initializer())

        def split_input(batch):
            x_batch = batch
            y_batch = np.concatenate([x_batch[:, 1:], np.zeros(shape=[x_batch.shape[0], 1], dtype=np.int32) + 36], axis=1)
            return x_batch, y_batch

        def train_step(x_batch):
            x_bat, y_bat = split_input(x_batch)
            feed_dict = {
                cnn.encoder_input: y_bat,
                cnn.encoder_input_GO: x_bat,
                cnn.gaussian_samples: np.random.normal(size=[x_batch.shape[0], FLAGS.unit_gaussian_dim]),
            }

            outputs = sess.run(
                [train_op, global_step, train_summary_op, cnn.total_loss, cnn.accuracy, cnn.mean_latent_loss, cnn.CE_loss],
                feed_dict)

            _, step, summaries, loss, accuracy, kldiv, CEloss = outputs

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, latentLoss: {:g}, reconstructionLoss: {:g}, acc {:g}".format(
                time_str, step, loss, kldiv,CEloss, accuracy))

            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, writer=None):
            x_bat, y_bat = split_input(x_batch)
            feed_dict = {
                cnn.encoder_input: y_bat,
                cnn.encoder_input_GO: x_bat,
                cnn.gaussian_samples: np.random.normal(size=[x_batch.shape[0], FLAGS.unit_gaussian_dim]),
            }

            step, summaries, loss, accuracy, kldiv, CEloss = sess.run(
                [global_step, dev_summary_op, cnn.total_loss, cnn.accuracy, cnn.mean_latent_loss, cnn.CE_loss],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, klDiv: {:g}, CE-loss: {:g}, acc {:g}".format(time_str, step, loss, kldiv,
                                                                                        CEloss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = preProcess.batch_iter(
            list(x_train), FLAGS.batch_size, FLAGS.num_epochs)
        for idx, batch in enumerate(batches):
            x_batch = batch
            current_step = tf.train.global_step(sess, global_step)
            train_step(x_batch)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

