import json
import tensorflow.compat.v1 as tf
import datasets
import model
import numpy as np

tf.disable_eager_execution()

print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('hidden_size', 512, 'Size of the LSTM hidden state.')
tf.flags.DEFINE_integer('embedding_size', 128, 'Size of word embeddings.')

tf.flags.DEFINE_string('save_dir', 'saved_models/model3/model', 'location to saved model')

tf.flags.DEFINE_float("learning_rate", 0.001 , "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-8, 'Epsilon for gradient update formula.')
tf.flags.DEFINE_float('max_grad_norm', 1, 'Maxmimum gradient norm.')

tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer('num_epochs', 10, 'number of training epochs.')
tf.flags.DEFINE_integer('num_curriculums', 4, 'number of training curriculums.')

tf.flags.DEFINE_integer("min_training_iterations", 1250,
                        "Minimum number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 25,
                        "Iterations between reports (samples, valid loss).")

best_dev_loss = 20

def get_datasets(group):
    global train_dataset, num_train_batches
    global dev_dataset, num_dev_batches
    global train_iterator, next_train_element
    global dev_iterator, next_dev_element

    train_dataset, num_train_batches = datasets.import_dataset('train', FLAGS.batch_size, group)
    train_iterator = tf.data.make_initializable_iterator(train_dataset)
    next_train_element = train_iterator.get_next()

    dev_dataset, num_dev_batches = datasets.import_dataset('dev', FLAGS.batch_size, group)
    dev_iterator = tf.data.make_initializable_iterator(dev_dataset)
    next_dev_element = dev_iterator.get_next()

def select_row_elements(mat, idx):
    final_shape = tf.shape(idx)
    row_size = mat.shape[-1]
    flat_mat = tf.reshape(mat, [-1])
    flat_idx = tf.reshape(idx, [-1])
    flat_idx += tf.range(0, tf.shape(flat_idx)[0]) * row_size
    return tf.reshape(tf.gather_nd(flat_mat, tf.expand_dims(flat_idx, axis=-1)), final_shape)

def build_model():
    global inp_placeholder, target_placeholder
    global logits, loss, avg_log_prob
    global grad_descent, global_step

    inp_placeholder = tf.placeholder(tf.int32, shape=(None, datasets.inp_size))
    target_placeholder = tf.placeholder(tf.int32, shape=(None, datasets.outp_size))

    logits, syntax_adjs = model.model(inp_placeholder, target_placeholder, FLAGS.hidden_size, FLAGS.embedding_size) # [batch, outp_size, num_tokens + 1]
    token_log_probabilities = tf.log(1e-10 + select_row_elements(tf.nn.softmax(logits), target_placeholder))
    syntax_adjs = select_row_elements(syntax_adjs, target_placeholder)
    avg_log_prob = tf.reduce_mean(token_log_probabilities)
    loss = -avg_log_prob - tf.reduce_mean(syntax_adjs)
    
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, trainable_variables), FLAGS.max_grad_norm)

    global_step = tf.get_variable(
            name='global_step',
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.AdamOptimizer(
        FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
    grad_descent = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step)

def train_iter(sess, curriculum, epoch, num_epochs, saver):
    sess.run(train_iterator.initializer)
    for batch in range(num_train_batches):
        inp, target = sess.run(next_train_element)
        batch_loss, log_prob, _ = sess.run([loss, avg_log_prob, grad_descent],
                feed_dict={inp_placeholder: inp, target_placeholder: target})
        if np.isnan(batch_loss):
            return False
        step = sess.run(global_step)
        if (batch + 1) % FLAGS.report_interval == 0:
            print('[Curriculum %d/%d][Epoch %d/%d] Training Iteration %d/%d : Loss = %.3f Avg_Token_Prob = %.2f%%'%(curriculum, FLAGS.num_curriculums, epoch + 1, num_epochs, batch + 1, num_train_batches, batch_loss, 10 ** (2 + log_prob)), flush=True)
        if step % 1000 == 0:
            dev_iter(step, sess, saver)
    return True

def dev_iter(step, sess, saver):
    global best_dev_loss

    total_loss = 0
    total_cnt = 0
    sess.run(dev_iterator.initializer)
    for batch in range(num_dev_batches):
        inp, target = sess.run(next_dev_element)
        batch_loss, log_prob = sess.run([loss, avg_log_prob],
                feed_dict={inp_placeholder: inp, target_placeholder: target})
        total_loss += batch_loss * inp.shape[0]
        total_cnt += inp.shape[0]
        if (batch + 1) % FLAGS.report_interval == 0:
            print('\tDev Iteration %d/%d : Loss = %.3f Avg_Token Prob = %.2f%%'%(batch + 1, num_dev_batches, batch_loss, 10 ** (2 + log_prob)), flush=True)

    curr_loss = total_loss / total_cnt

    print('\tAverage Dev Loss = %.3f'%(curr_loss))

    if curr_loss < best_dev_loss:
        best_dev_loss = curr_loss
        print('\t** Current Model Saved')
        saver.save(sess, FLAGS.save_dir, global_step=step)

def train_model(min_training_iterations):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
        for i in range(1, FLAGS.num_curriculums):
            get_datasets(i)
            with tf.device('/device:GPU:0'):
                train_iter(sess, i, 0, 1, saver)
        get_datasets(FLAGS.num_curriculums)

        epoch = 0
        while epoch < FLAGS.num_epochs:
            with tf.device('/device:GPU:0'):
                train_iter(sess, FLAGS.num_curriculums, epoch, FLAGS.num_epochs, saver)
            """
            if not train_iter(sess, epoch, saver):
                print('restarting training')
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                epoch = 0
            else:
                epoch += 1"""
            epoch += 1

def main():
    build_model()
    tf.config.set_soft_device_placement(True)
    train_model(FLAGS.min_training_iterations)

if __name__ == '__main__':
    exit(main())
