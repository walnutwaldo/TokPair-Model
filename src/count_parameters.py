import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('save_dir', 'saved_models/model1-6000', 'location to saved model')

def main():
    tf.compat.v1.train.import_meta_graph(FLAGS.save_dir + '.meta')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.save_dir)
        total_params = 0
        for v in tf.trainable_variables():
            print('%s %s %d'%(v.name, str(v.get_shape()), np.product(v.get_shape())))
            total_params += np.product(v.get_shape())
        print(str(total_params) + ' parameters')

if __name__ == '__main__':
    exit(main())
