import tensorflow.compat.v1 as tf
import datasets
import math

def linear(inp, outp_size, activation, name):
    with tf.variable_scope(name + '_linear', reuse=tf.AUTO_REUSE):
        inp_size = inp.shape[-1]
        w = tf.get_variable('weight',
                shape=[inp_size, outp_size],
                initializer=tf.initializers.variance_scaling)
        b = tf.get_variable('bias',
                shape=[outp_size],
                initializer=tf.constant_initializer(0))
        res = tf.matmul(inp, w) + b
        if activation is not None:
            res = activation(res)
        return res


def lstm(inp, ltm, stm, hidden_size, name):
    with tf.variable_scope(name + '_lstm', reuse=tf.AUTO_REUSE):
        seq_len = inp.shape[1]
        
        if ltm is None:
            ltm = tf.get_variable('initial_ltm',
                    shape = [1, hidden_size],
                    initializer=tf.random_normal_initializer(stddev=1))
            ltm = tf.tile(ltm, [tf.shape(inp)[0], 1])
        if stm is None:
            stm = tf.get_variable('initial_stm',
                    shape = [1, hidden_size],
                    initializer=tf.random_uniform_initializer(minval=0, maxval=1))
            stm = tf.tile(stm, [tf.shape(inp)[0], 1])

        inp = tf.transpose(inp, [1, 0, 2])
        outps = []
        for t in range(seq_len):
            inp_tokens = inp[t]
            h = tf.concat([stm, inp_tokens], axis=1)

            forget = linear(h, hidden_size, tf.sigmoid, 'forget')
            ltm *= forget

            write = linear(h, hidden_size, tf.tanh, 'write')
            write *= linear(h, hidden_size, tf.sigmoid, 'write_filter')
            ltm += write

            outp = linear(h, hidden_size, tf.sigmoid, 'outp') * ltm
            outps.append(outp)

    return tf.transpose(tf.stack(outps, axis=0), [1, 0, 2]), ltm

def attn(targets, keys, outp_size):
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        w = tf.nn.softmax(tf.matmul(targets, keys, transpose_b=True))
        c = tf.matmul(w, keys)
        return linear(tf.concat([targets, c], axis=2), outp_size, None, 'combine')

def model(inp, target, hidden_size, embedding_size):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        # embedding
        vocab_embedding = tf.get_variable('vocab_embedding',
                shape=[datasets.vocab_size + 1, embedding_size],
                initializer=tf.random_normal_initializer(stddev=1))
        token_embedding = tf.get_variable('token_embedding',
                shape=[datasets.num_tokens + 1, embedding_size],
                initializer=tf.random_normal_initializer(stddev=1))

        inp = tf.gather_nd(vocab_embedding, tf.expand_dims(inp, axis=-1))
        target = tf.gather_nd(token_embedding, tf.expand_dims(target, axis=-1))
        shifted_target = tf.get_variable('decoder_start_token', shape=[1, 1, embedding_size], initializer=tf.random_normal_initializer(stddev=1))
        shifted_target = tf.tile(shifted_target, [tf.shape(inp)[0], 1, 1])
        shifted_target = tf.concat([shifted_target, target], axis=1)[:,:-1,:]

        # encoding 
        encoder_outps, encoder_ltm = lstm(inp, None, None, hidden_size, 'encoding')

        # decoding
        last_encoder_outp = encoder_outps[:,-1,:]
        decoder_outps, _ = lstm(shifted_target, encoder_ltm, last_encoder_outp, hidden_size, 'decoding')
    
        # attention
        outp = attn(decoder_outps, encoder_outps, embedding_size)
        outp = tf.matmul(outp, token_embedding, transpose_b=True)

        #syntax
        syntax_mask, _ = lstm(shifted_target, None, None, hidden_size, 'syntax')
        syntax_mask = linear(syntax_mask, datasets.num_tokens + 1, tf.exp, 'syntax')

        outp = outp - syntax_mask

        return outp, syntax_mask
