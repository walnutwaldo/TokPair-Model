import json
import inspect

from program_synthesis.algolisp.dataset.code_lisp import *
import program_synthesis.algolisp.dataset.code_types as code_types
from program_synthesis.algolisp.dataset.code_lisp_parser import parse_and_compile
from program_synthesis.algolisp.dataset import code_trace

def get_prob_data(file_name, probnum):
    with open(file_name, 'r') as f:
        line_cnt = 0
        while line_cnt != probnum:
            f.readline()
            line_cnt += 1
        line = f.readline()
    return json.loads(line)

if __name__ == '__main__':
    probdata = get_prob_data('data/metaset3.test.jsonl', 5)
    print(' '.join(probdata['text']))
    args = sorted(list(probdata['args'].items()))
    prog = probdata['short_tree']
    lisp_units = load_lisp_units() # the types of nodes (except != which is combining 2)
    f = compile_func(lisp_units, 'testfunc', prog, args,
            code_types._TYPES[probdata['return_type']])
    print(f([1, 2, 5, 3, 4, 2], 4))

