import math
import copy

debug = False

def is_prime(x):
    if x < 2:
        return False
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True

def is_sorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True

def get_digits(x):
    x = abs(x)
    return [int(c) for c in str(x)]

def get_range(lo, hi):
    if hi - lo > 10000:
        raise RuntimeError('Asking for too large of range')
    return [x for x in range(lo, hi)]

def make_partial1(v, f):
    return lambda x: f(x, v)

def make_partial0(v, f):
    return lambda x: f(v, x)

def str_split(s, t):
    _res = s.split(t)
    res = []
    for a in _res:
        if a:
            res.append(a)
    return res

def deref(arr, idx):
    if idx < 0 or idx >= len(arr):
        raise RuntimeError("index out of bounds")
    return arr[idx]

operator_dict = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x // y,
        '%': lambda x, y: x % y,
        'int-deref': deref,
        'sqrt': math.sqrt,
        'floor': int,
        'min': min,
        'max': max, 
        'str_index': lambda x, y: deref(y, x),
        'str_concat': lambda s, t: s + t,
        'str_min': min,
        'str_max': max,
        'str_split': str_split,
        'strlen': len,
        'len': len,
        'head': lambda x : deref(x, 0),
        'sort': sorted,
        'digits': get_digits,
        'range': get_range,
        '!': lambda x: not x,
        '==': lambda x, y: x == y,
        '||': lambda x, y: x or y,
        '&&': lambda x, y: x and y,
        '>': lambda x, y: x > y,
        '<': lambda x, y: x < y,
        '>=': lambda x, y: x >= y,
        '<=': lambda x, y: x <= y,
        'square': lambda x: x * x,
        'is_prime': is_prime,
        'reverse': lambda x : list(reversed(x)),
        'slice': lambda x, lo, hi: x[lo:hi],
        'partial0': make_partial0,
        'partial1': make_partial1,
        'combine': lambda f, g: (lambda x: f(g(x))),
        'deref': deref,
        'is_sorted': is_sorted,
        }

def memoize(f):
    mem = {}
    def helper(x):
        if x not in mem:
            mem[x] = f(x)
        return mem[x]
    return helper

def create_lambda1(subprog, args):
    @memoize
    def f(x):
        newargs = copy.deepcopy(args)
        newargs['arg1'] = x
        newargs['self'] = f
        return run_program(subprog, newargs)
    return f

def create_lambda2(subprog, args):
    def f(x, y):
        newargs = copy.deepcopy(args)
        newargs['arg1'] = x
        newargs['arg2'] = y
        newargs['self'] = f
        return run_program(subprog, newargs)
    return f

def run_program(program, args):
    res = _run_program(program, args)
    if debug:
        print('running ' + str(program) + ' on ' + str(args) + ' is ' + str(res))
    return res

def _run_program(program, args):
    if type(program) is list:
        if program[0] in operator_dict:
            f = operator_dict[program[0]]
            inps = [run_program(p, args) for p in program[1:]]
            return f(*inps)
        if program[0] == 'invoke1':
            f = run_program(program[1], args)
            return f(run_program(program[2], args))
        if program[0] == 'lambda1':
            return create_lambda1(program[1], args)
        if program[0] == 'lambda2':
            return create_lambda2(program[1], args)
        if program[0] == 'if':
            cond = run_program(program[1], args)
            if cond:
                return run_program(program[2], args)
            else:
                return run_program(program[3], args)
        if program[0] == "self":
            f = args['self']
            return f(run_program(program[1], args))
        if program[0] == 'reduce':
            arr = run_program(program[1], args)
            iv = run_program(program[2], args)
            f = run_program(program[3], args)
            for a in arr:
                iv = f(iv, a)
            return iv
        if program[0] == 'filter':
            arr = run_program(program[1], args)
            f = run_program(program[2], args)
            res = []
            for a in arr:
                if f(a):
                    res.append(a)
            return res
        if program[0] == "map":
            arr = run_program(program[1], args)
            f = run_program(program[2], args)
            return [f(a) for a in arr]

        print('idk what ' + program[0] + ' is')
    else:
        if program in args:
            return args[program]
        try:
            x = int(program)
            return x
        except:
            pass
        if program[0] == '"':
            return program[1:-1]
        if program in ['true', 'false']:
            return (True if program == 'true' else False)
        if program in operator_dict:
            return operator_dict[program]
        print('uhhh idk what ' + program + ' is')
