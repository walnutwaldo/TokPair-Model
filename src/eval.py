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
tf.flags.DEFINE_integer("beam_size", 1, "beam size of search.")

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

def build_onto_model():
    global encoder_outps, encoder_ltm
    global encoder_outps_ph, encoder_ltm_ph
    global prev_token_ph, pass_data_ph_map
    global decode_next, decoder_pass_data
    global init_decoder_outp, init_decoder_pass_data

    encoder_outps, encoder_ltm = model.run_encoder(inp_placeholder, FLAGS.hidden_size, FLAGS.embedding_size)

    prev_token_ph = tf.placeholder(tf.int32, shape=(None))
    encoder_outps_ph = tf.placeholder(tf.float32, shape=(None, datasets.inp_size, FLAGS.hidden_size))
    encoder_ltm_ph = tf.placeholder(tf.float32, shape=(None, FLAGS.hidden_size))

    init_decoder_outp, init_decoder_pass_data = model.run_single_decoder([datasets.num_tokens + 1], encoder_outps_ph[:,-1,:], encoder_ltm_ph, None, None, encoder_outps_ph, FLAGS.hidden_size, FLAGS.embedding_size)

    pass_data_ph_map = {}
    pass_data_ph_map['last_decoder_outp_ph'] = tf.placeholder(tf.float32, shape=(None, FLAGS.hidden_size))
    pass_data_ph_map['last_decoder_ltm_ph'] = tf.placeholder(tf.float32, shape=(None, FLAGS.hidden_size))
    pass_data_ph_map['last_syntax_outp_ph'] = tf.placeholder(tf.float32, shape=(None, FLAGS.hidden_size))
    pass_data_ph_map['last_syntax_ltm_ph'] = tf.placeholder(tf.float32, shape=(None, FLAGS.hidden_size))

    decode_next, decoder_pass_data = model.run_single_decoder(prev_token_ph,
            pass_data_ph_map['last_decoder_outp_ph'],
            pass_data_ph_map['last_decoder_ltm_ph'],
            pass_data_ph_map['last_syntax_outp_ph'],
            pass_data_ph_map['last_syntax_ltm_ph'],
            encoder_outps_ph,
            FLAGS.hidden_size, FLAGS.embedding_size)

def softmax(x):
    x = x - np.amax(x, axis=-1)
    y = np.exp(x)
    return y / np.sum(y, axis=-1)

def try_problem(problem, sess):
    text = np.array([datasets.extend(problem['encoded_text'], 
            datasets.inp_size, datasets.vocab_size)])
    target_program = np.array([datasets.extend(datasets.flatten(problem['encoded_tree']),
            datasets.outp_size, datasets.num_tokens)])

    this_loss, this_log_prob = sess.run([loss, avg_log_prob],
            feed_dict={inp_placeholder:text, target_placeholder: target_program})

    this_encoder_outps, this_encoder_ltm = sess.run([encoder_outps, encoder_ltm],
            feed_dict={inp_placeholder: text})

    outp, pass_data = sess.run([init_decoder_outp, init_decoder_pass_data],
            feed_dict={encoder_outps_ph: this_encoder_outps, encoder_ltm_ph: this_encoder_ltm})
    prob_dist = softmax(outp)

    queue = [(np.log(1e-10 + prob_dist[0, i]), [i], pass_data) for i in range(datasets.num_tokens + 1)]
    queue = sorted(queue, key=lambda x: x[0])[-FLAGS.beam_size:]

    while True:
        program, base_prob, pass_data = None, None, None
        for i in range(len(queue) - 1, -1, -1):
            if queue[i][1][-1] != datasets.num_tokens and len(queue[i][1]) < datasets.outp_size:
                program = queue[i][1]
                base_prob = queue[i][0]
                pass_data = queue[i][2]
                queue = queue[:i] + queue[i + 1:]
                break
        if program is None:
            break

        outp, pass_data = sess.run([decode_next, decoder_pass_data], feed_dict={
                prev_token_ph: np.asarray([program[-1]]),
                pass_data_ph_map['last_decoder_outp_ph']: pass_data[0],
                pass_data_ph_map['last_decoder_ltm_ph']: pass_data[1],
                pass_data_ph_map['last_syntax_outp_ph']: pass_data[2],
                pass_data_ph_map['last_syntax_ltm_ph']: pass_data[3],
                encoder_outps_ph: this_encoder_outps})
        prob_dist = softmax(outp)
        new_states = [(np.log(1e-10 + prob_dist[0, i]) + base_prob, program + [i], pass_data) \
                for i in range(datasets.num_tokens + 1)]
        queue = sorted(queue + new_states, key=lambda x: x[0])[-FLAGS.beam_size:]
    for i in range(FLAGS.beam_size):
        solved = True
        for a, b in zip(target_program[0], queue[i][1]):
            if int(a) == datasets.num_tokens:
                break
            if int(a) != int(b):
                solved = False
        if solved:
            return this_loss, this_log_prob, True

    return this_loss, this_log_prob, False

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
            if (i + 1) % 1 == 0:
                print('Evaluation [%d/%d]\n\tLoss: %.3f\n\tAverage Token Probability = %.2f%%\n\tPercent Solved: %.2f%%'%(i + 1, len(eval_problems), total_loss/(i + 1), 10 ** (total_log_prob/(i + 1) + 2), total_solved/(i + 1) * 100))
        print('Final Evaluation\n\tLoss: %.3f\n\tAverage Token Probability = %.2f%%\n\tPercent Solved: %.2f%%'%(total_loss/len(eval_problems), 10 ** (total_log_prob/len(eval_problems) + 2), total_solved/len(eval_problems) * 100))

def main():
    get_datasets()
    build_model()
    build_onto_model()
    eval_saved_model()

if __name__ == '__main__':
    exit(main())
