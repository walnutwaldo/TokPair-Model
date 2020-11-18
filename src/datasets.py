import tensorflow.compat.v1 as tf
import json
import numpy as np

data_group = 'filtered_data2' # folder to look in for data

num_tokens = 150 # this is the first number provided to encode.py (does not include pad)
vocab_size = 284 # analyze.py will provide the correct number followed by "words" (does not include pad)

inp_size = 164 # analyze.py will provide the correct number preceded by "longest text:"
outp_size = 109 # analyze.py will provide the correct number preceded by "longest program:"

shuffle_buffer = 1000

def flatten(x):
    if type(x) is list:
        res = []
        for a in x:
            res.extend(flatten(a))
        return res
    else:
        return [x]

def extend(x, new_len, pad):
    return x + [pad] * (new_len - len(x))

def import_dataset(dset_type, batch_size, group=None):
    file_name = data_group + '/encoded/%s-%d%s.jsonl'%(dset_type, num_tokens, '-' + str(group) if group else '')
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

def import_raw_dataset(dset_type, group=None):
    file_name = data_group + '/encoded/%s-%d%s.jsonl'%(dset_type, num_tokens, '-' + str(group) if group else '')
    print('loading %s ... '%file_name, end='')
    problems = []
    with open(file_name, 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems.append(problem)
    print('DONE')
    return problems
