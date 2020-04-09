import json
import numpy as np
import tensorflow.compat.v1 as tf
import datasets
import model
import random

tf.disable_eager_execution()

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('hidden_size', 512, 'Size of the LSTM hidden state.')
tf.flags.DEFINE_integer('embedding_size', 128, 'Size of word embeddings.')

tf.flags.DEFINE_string('save_dir', 'saved_models/model2/model.ckpt-13000', 'location to saved model')
tf.flags.DEFINE_float("learning_rate", 0.001 , "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-8, 'Epsilon for gradient update formula.')
tf.flags.DEFINE_float('max_grad_norm', 1, 'Maxmimum gradient norm.')

tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")

tf.flags.DEFINE_integer("report_interval", 25,
                        "Iterations between reports (samples, valid loss).")

def get_datasets():
    global eval_problems
    eval_problems = datasets.import_raw_dataset('test')
    random.shuffle(eval_problems)

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

def try_problem(problem, sess):
    text = np.array([datasets.extend(problem['encoded_text'], datasets.inp_size, datasets.vocab_size)])
    target_program = np.array([datasets.extend(datasets.flatten(problem['encoded_tree']), datasets.outp_size, datasets.num_tokens)])

    _loss, _log_prob = sess.run([loss, avg_log_prob], feed_dict={inp_placeholder:text, target_placeholder: target_program})

    return _loss, _log_prob, True

def eval_saved_model():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.save_dir)
        total_loss = 0
        total_log_prob = 0
        total_solved = 0
        for i, problem in enumerate(eval_problems):
            _loss, _log_prob, solved = try_problem(problem, sess)
            if solved:
                total_solved += 1
            total_loss += _loss
            total_log_prob += _log_prob
            if (i + 1) % 100 == 0:
                print(_loss, _log_prob)
                print('Evaluation [%d/%d]\n\tLoss: %.3f\n\tAverage Token Probability = %.2f%%\n\tPercent Solved: %.2f'%(i + 1, len(eval_problems), total_loss/(i + 1), 10 ** (total_log_prob/(i + 1) + 2), total_solved/(i + 1) * 100))
        print('Final Evaluation\n\tLoss: %.3f\n\tAverage Token Probability = %.2f%%\n\tPercent Solved: %.2f'%(total_loss/len(eval_problems), 10 ** (total_log_prob/len(eval_problems) + 2), total_solved/len(eval_problems * 100)))

def main():
    get_datasets()
    build_model()
    eval_saved_model()

if __name__ == '__main__':
    exit(main())
