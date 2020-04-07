import tensorflow.compat.v1 as tf
import json
import numpy as np

filtered = True
num_tokens = 81 # does not include pad
vocab_size = 284 # dows not include pad

inp_size = 164
outp_size = 217

shuffle_buffer = 1000

def flatten(x):
    res = []
    for a in x:
        if type(a) is list:
            res.extend(flatten(a))
        else:
            res.append(a)
    return res

def extend(x, new_len, pad):
    return x + [pad] * (new_len - len(x))

def import_dataset(dset_type, batch_size):
    file_name = 'data/encoded/%s-%d.jsonl'%(dset_type, num_tokens)
    if filtered:
        file_name = 'filtered_' + file_name
    print('loading %s ... '%file_name, end='')
    problems = []
    with open(file_name, 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems.append(problem)
    texts = np.array([extend(problem['encoded_text'], inp_size, vocab_size) for problem in problems], dtype=np.int32)
    programs = np.array([extend(flatten(problem['encoded_tree']), outp_size, num_tokens) for problem in problems], dtype=np.int32)
    #tests = [problem['tests'] for problem in problems]

    dataset = tf.data.Dataset.from_tensor_slices((texts, programs))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    
    print('DONE')
    return dataset, (texts.shape[0] - 1) // batch_size + 1
