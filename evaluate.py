import tensorflow as tf
import numpy as np
import pickle
import model
import preProcess

tf.flags.DEFINE_float("dev_sample_percentage", .03, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "data/drugVectorsGo.pickle", "Data source.")
tf.flags.DEFINE_string("parameters_file", "runs/1518484761/checkpoints/model-45000", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("vocab_size", 37, "number of chars in SMILES vocab)")
tf.flags.DEFINE_integer("max_molecule_length", 50, "number of chars in SMILES vocab)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5,6", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda")
tf.flags.DEFINE_float("unit_gaussian_dim", 300, "number of gaussians")

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

print("\nEvaluating...\n")


def split_input(batch):
    x_batch = batch
    y_batch = np.concatenate([x_batch[:, 1:], np.zeros(shape=[x_batch.shape[0], 1], dtype=np.int32) + 36], axis=1)
    return x_batch, y_batch


def get_smile(ar, num2char):
    smile = ""
    for i in ar:
        smile += num2char[str(i)]
    return smile


def analyze_output(predictions):
    # analyze output
    num2char = preProcess.load_json_file('data/num2char.json')
    num2char['34'] = 'GO' ; num2char['35'] = 'EN' ; num2char['36'] = 'PA'
    count = 0
    new_mol = set()
    for index, pred in enumerate(predictions):
        real_smile = get_smile(x_bat[index], num2char)
        real_smile = real_smile.split('EN')[0].split('GO')[1]
        fake_smile = get_smile(pred, num2char)
        fake_smile = fake_smile.split('EN')[0]
        res = preProcess.mol_analysis(fake_smile, real_smile)
        count += res
        if res == 1:
            if fake_smile != real_smile:
                new_mol.add(fake_smile)
    return new_mol, count

# Evaluation
# ==================================================
graph = tf.Graph()
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        CDN = model.CDN(
            sequence_length=x_train.shape[1],
            vocab_size=FLAGS.vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            max_molecule_length=FLAGS.max_molecule_length,
            gaussian_samples=FLAGS.unit_gaussian_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            variational=True,
            test_mode=True,
        )

        encoded, latent_loss = CDN.encode()
        logits = CDN.decode_rnn(encoded)
        loss, accuracy = CDN.loss(logits, latent_loss)

        checkpoint_file = FLAGS.parameters_file
        saver = tf.train.Saver().restore(sess, checkpoint_file)
        print "restored " + str(checkpoint_file)

        x_bat, y_bat = split_input(x_dev)
        feed_dict = {
            CDN.encoder_input: y_bat,
            CDN.encoder_input_GO: x_bat,
            CDN.gaussian_samples: np.random.normal(size=[x_bat.shape[0], FLAGS.unit_gaussian_dim], scale=1.0),
        }

        outputs = sess.run(
            [encoded, latent_loss, logits, loss, accuracy, CDN.all_symbols], feed_dict=feed_dict)

        predictions = np.argmax(outputs[2], axis=2)
        #predictions = outputs[-1]

        new_mol, valid = analyze_output(predictions)
        print "Loss " + str(outputs[3])
        print "Acc with respect to real: " + str(outputs[4])
        print "Total valid molecules: " + str(valid/float(len(predictions)))
        print "New mols: " + str(len(new_mol))




