import json
import copy
from collections import Counter
from program_synthesis.algolisp.dataset import executor

vocab_size = 100
primitives = []
sketches = []

def load_sketches():
    with open('sketches.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'par_sketch' in data:
                sketches.append(data)
            else:
                primitives.append(data)

            if len(sketches) + len(primitives) == vocab_size:
                break

def primitive_encode(tree):
    token_set = set()
    if type(tree) is list:
        for p in primitives:
            if type(p['tree']) is list and p['tree'][0] == tree[0]:
                token_set.add(p['id'])
                res = [p['id']]
                for a in tree[1:]:
                    t, ts = primitive_encode(a)
                    res.append(t)
                    token_set = token_set.union(ts)
                return res, token_set
    else:
        for p in primitives:
            if type(p['tree']) is not list and p['tree'] == tree:
                token_set.add(p['id'])
                return p['id'], token_set

def apply_sketch(sketch, tree, token_set):
    def recur(prog):
        return apply_sketch(sketch, prog, token_set)

    if type(tree) is not list:
        return tree

    if tree[0] == sketch['par_sketch'] and len(tree) > 1 + sketch['child_idx']:
        x = tree[sketch['child_idx'] + 1]
        if type(x) is list and x[0] == sketch['child_sketch']:
            tree = [sketch['id']] + tree[1:sketch['child_idx'] + 1] + x[1:] + tree[sketch['child_idx'] + 2:]
            token_set.add(sketch['id'])
        elif type(x) is not list and x == sketch['child_sketch']:
            tree = [sketch['id']] + tree[1:sketch['child_idx'] + 1] + tree[sketch['child_idx'] + 2:]
            token_set.add(sketch['id'])

    if len(tree) is 1:
        return tree[0]
    return [recur(p) for p in tree]

def encode_tree(tree):
    res, token_set = primitive_encode(tree)
    for sketch in sketches:
        if sketch['par_sketch'] in token_set and sketch['child_sketch'] in token_set:
            res = apply_sketch(sketch, res, token_set)
    return res

def encode(in_file, out_file):
    problems = []
    with open(in_file, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    with open(out_file, 'w') as f:
        for problem in problems:
            problem['encoded_tree'] = encode_tree(problem['short_tree'])
            json.dump(problem, f)
            f.write('\n')

def main():
    load_sketches()
    for dset in 'train dev test'.split():
        encode('metaset3.' + dset + '.jsonl', 'encoded/' + dset + '-' + str(vocab_size) + '.jsonl')

if __name__ == '__main__':
    exit(main())
