import json
import copy
from collections import Counter
from program_synthesis.algolisp.dataset import executor

vocab_size = 200

sketches = []

programs = []
command_cnt = Counter()
command_pair_cnt = Counter()

def deep_equals(l1, l2):
    if not type(l1) == type(l2):
        return False
    if type(l1) is list:
        if len(l1) != len(l2):
            return False
        for a, b in zip(l1, l2):
            if not deep_equals(a, b):
                return False
        return True
    else:
        return l1 == l2

def index_of_sketch(sketch):
    for i, possibility in enumerate(sketches):
        if deep_equals(sketch, possibility):
            return i
    return -1

def encode(program):
    if type(program) is list:
        sketch = [program[0]] + ['<HOLE>'] * (len(program) - 1)
        program[0] = index_of_sketch(sketch)
        if program[0] == -1:
            sketches.append(sketch)
            program[0] = len(sketches) - 1

        for i in range(1, len(program)):
            program[i] = encode(program[i])
    else:
        if program not in sketches:
            sketches.append(program)
        program = sketches.index(program)
    return program

def combine(par_sketch, child_sketch, child_idx):
    if par_sketch == '<HOLE>':
        if child_idx == 0:
            return child_sketch
        else:
            return child_idx - 1
    elif type(par_sketch) is not list:
        return child_idx
    else:
        for i in range(1, len(par_sketch)):
            rec = combine(par_sketch[i], child_sketch, child_idx)
            if type(rec) is int:
                child_idx = rec
            else:
                res = [par_sketch[j] if j != i else rec for j in range(len(par_sketch))]
                return res
        return child_idx

def add_programs(file_name):
    print('processing %s ... ' % file_name, end='\r', flush=True)
    line_cnt = 0
    with open(file_name, 'r') as f:
        for line in f:
            line_cnt += 1
            data = json.loads(line)
            program = data['short_tree']
            program = encode(program)
            programs.append(program)
            if (line_cnt % 100 == 0):
                print('processing %s ... %d' % (file_name, line_cnt), end='\r', flush=True)
    print('processing %s ... DONE  ' % file_name, flush=True)

def init_cnt(program):
    if type(program) is not list:
        command_cnt[program] += 1
        return
    command_cnt[program[0]] += 1
    for i, a in enumerate(program[1:]):
        child_command = a[0] if type(a) is list else a
        command_pair_cnt[(program[0], i, child_command)] += 1 
        init_cnt(a)

def upd(program, par_sketch, child_sketch, child_idx, new_sketch):
    def recur(prog):
        return upd(prog, par_sketch, child_sketch, child_idx, new_sketch)

    if type(program) is not list:
        return program

    if program[0] == par_sketch and child_idx + 1 < len(program):
        x = program[child_idx + 1]
        if type(x) is list:
            if x[0] == child_sketch:
                program = [new_sketch] + program[1:child_idx + 1] + x[1:] + program[child_idx + 2:]
        else:
            if x == child_sketch:
                program = [new_sketch] + program[1:child_idx + 1] + program[child_idx + 2:]
        
    if len(program) is 1:
        return program[0]
    return [recur(x) for x in program]

def init():
    for program in programs:
        init_cnt(program)

def main():
    for t in 'train dev test'.split():
        add_programs('metaset3.' + t + '.jsonl')
    init()

    with open('sketches.jsonl', 'w') as f:
        for i, s in enumerate(sketches):
            json.dump({'id':i, 'tree':s, 'frequency':command_cnt[i]}, f)
            f.write('\n')

    combinations = []
    while len(sketches) < vocab_size:
        (par_sketch, child_idx, child_sketch), freq = command_pair_cnt.most_common(1)[0]
        print(freq)
        print('combining %s and %s at index %d' % (str(sketches[par_sketch]), str(sketches[child_sketch]), child_idx), flush=True)
        sketches.append(combine(sketches[par_sketch], sketches[child_sketch], child_idx))
        command_pair_cnt.clear()
        for i in range(len(programs)):
            programs[i] = upd(programs[i], par_sketch, child_sketch, child_idx, len(sketches) - 1)
            init_cnt(programs[i])
            print('%d/%d' % (i + 1, len(programs)), end='\r', flush=True)
        print()
        combinations.append({'id':len(sketches) - 1, 'tree':sketches[-1], 'par_sketch':par_sketch, 'child_sketch':child_sketch, 'child_idx':child_idx, 'frequency':freq})

    with open('sketches.jsonl', 'a') as f:
        for c in combinations:
            json.dump(c, f)
            f.write('\n')

if __name__ == '__main__':
    exit(main())
