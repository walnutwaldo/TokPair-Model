import copy

debug = False

literal_map = {"1" : "int", "0" : "int", "2" : "int", "10" : "int",
        "1000000000" : "int", "40" : "int", "true" : "bool", "false" : "bool",
        "+": "(int, int)->int", "/":"(int, int)->int", "*":"(int, int)->int",
        "%": "(int, int)->int", "int-deref":"(int[], int)->int",
        "!":"(bool)->bool", "is_prime":"(int)->bool", "square": "(int)->int",
        "==": "comparison", "||": "(bool, bool)->bool",
        ">": "comparison", "<": "comparison", ">=": "comparison", "<=": "comparison",
        "min": "(int, int)->int", "max": "(int, int)->int", '" "': "string",
        "str_index": "(int, string)->string", '""':"string", "str_concat":"(string, string)->string",
        "&&": "(bool, bool)->bool", '"z"': "string", "str_min": "(string, string)->string",
        "-":"(int, int)->int", "str_max": "(string, string)->string",
        "digits": "(int)->int[]", "range":"(int, int)->int[]",
        "str_split": "(string, string)->string[]", "sqrt": "(int)->float",
        "floor": "(float)->int", "strlen": "(string)->int"}

def func_params(f):
    if f is None:
        return None
    if f[0] != '(':
        return []
    return f.split(')->')[0][1:].split(', ')

def func_rt(f):
    if f is None:
        return None
    if f == "comparison":
        return "bool"
    if f[0] != '(':
        return f
    return f.split(')->')[1]

def is_array_type(ot):
    if ot is None:
        return False
    if ot[0] == '(':
        return False
    return ot[-2:] == '[]'

def expected_outp(program, args):
    res = _expected_outp(program, args)
    if debug:
        print('expected output of ' + str(program) + ' ' + str(args) + ' is ' + str(res))
    return res

def _expected_outp(program, args):
    if type(program) is list:
        if program[0] in literal_map:
            func_type = literal_map[program[0]]
            rt = func_rt(func_type)
            return rt
        if program[0] == "self":
            return None
        if program[0] in ["reduce", "map", "if"]:
            return expected_outp(program[2], args) 
        if program[0] in ["reverse", "slice", "sort", "filter", "invoke1", "lambda1", "lambda2"]:
            return expected_outp(program[1], args)
        if program[0] in ["deref", "head"]:
            ot = expected_outp(program[1], args)
            if not is_array_type(ot):
                return None
            return ot[:-2]
        if program[0] == "is_sorted":
            return "bool"
        if program[0] == "len":
            return "int"

    else:
        if program in args:
            return args[program]
        elif program in literal_map:
            return literal_map[program]
        return None

def outp_type(program, args, expected_params=[]):
    res = _outp_type(program, args, expected_params)
    if debug:
        print('outp type of ' + str(program) + ' ' + str(args) + ' ' + str(expected_params) + ' is ' + str(res))
    return res

def _outp_type(program, args, expected_params=[]):
    if type(program) is list:
        if program[0] in literal_map:
            functype = literal_map[program[0]]
            if functype == "comparison":
                t1 = outp_type(program[1], args)
                t2 = outp_type(program[2], args)
                if t1 is None or t2 is None:
                    return None
                if t1 == t2:
                    return "bool"
                return None
            params = func_params(functype) 
            if len(program) != len(params) + 1:
                return None
            for subprog, param_type in zip(program[1:], params):
                ot = outp_type(subprog, args)
                if ot is None or ot != param_type:
                    return None
            return func_rt(functype)
        if program[0] == "invoke1":
            ot = outp_type(program[2], args)
            if ot is None:
                return None

            functype = outp_type(program[1], args, [ot])
            params = func_params(functype)
            if params != [ot]:
                return None
            return func_rt(functype)
        if len(program[0]) > 6 and program[0][:6] == "lambda":
            cnt = int(program[0][-1])
            if len(expected_params) != cnt:
                return None
            newargs = copy.deepcopy(args)
            for i in range(cnt):
                newargs['arg' + str(i + 1)] = expected_params[i]
            exp_outp = expected_outp(program[1], newargs)
            if exp_outp is None:
                return None
            newargs['self'] = '(' + ', '.join(expected_params) + ')->' + exp_outp

            ot = outp_type(program[1], newargs)
            if ot != exp_outp:
                return None
            return newargs['self']
        if program[0] == "if":
            ot1 = outp_type(program[1], args)
            ot2 = outp_type(program[2], args)
            ot3 = outp_type(program[3], args)
            if ot1 is None or ot2 is None or ot3 is None:
                return None
            if ot1 == "bool" and ot2 == ot3:
                return ot2
            return None
        if program[0] == "self":
            if 'self' not in args:
                return None
            params = func_params(args['self'])
            if len(params) != 1:
                return None
            ot = outp_type(program[1], args)
            if ot is None:
                return None
            if ot == params[0]:
                return func_rt(args['self'])
            return None
        if program[0] == "reduce":
            ot1 = outp_type(program[1], args)
            ot2 = outp_type(program[2], args)
            if not is_array_type(ot1) or ot2 is None:
                return None
            ot3 = outp_type(program[3], args, [ot1[:-2], ot2])
            if ot3 is None:
                return None
            if func_params(ot3) != [ot2, ot1[:-2]]:
                return None
            if func_rt(ot3) != ot2:
                return None
            return ot2
        if program[0] == "slice":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            if outp_type(program[2], args) != "int" or outp_type(program[3], args) != "int":
                return None
            return ot1
        if program[0] == "map":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            func_type = outp_type(program[2], args, [ot1[:-2]])
            if func_params(func_type) == [ot1[:-2]]:
                if func_rt(func_type) is None:
                    return None
                return func_rt(func_type) + "[]"
            return None
        if len(program[0]) > 7 and program[0][:7] == "partial":
            idx = int(program[0][-1])
            ot1 = outp_type(program[1], args)
            if ot1 is None:
                return None
            if len(expected_params) < idx:
                return None
            ep = expected_params[:idx] + [ot1] + expected_params[idx:]
            functype = outp_type(program[2], args, ep)
            if func_params(functype) != ep:
                return None
            if func_rt(functype) is None:
                return None
            return '(' + ', '.join(expected_params) + ')->' + func_rt(functype)
        if program[0] == "reverse":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            return ot1
        if program[0] == "filter":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            func_type = outp_type(program[2], args, [ot1[:-2]])
            if func_params(func_type) == [ot1[:-2]]:
                if func_rt(func_type) != "bool":
                    return None
                return ot1
            return None
        if program[0] == "combine":
            ot1 = outp_type(program[2], args, expected_params)
            actual_params = func_params(ot1)
            if actual_params != expected_params:
                return None
            rt = func_rt(ot1)
            if rt is None:
                return None

            ot2 = outp_type(program[1], args, [rt])
            actual_params = func_params(ot2)
            if actual_params != [rt]:
                return None
            rt = func_rt(ot2)
            if rt is None:
                return None
            return '(' + ', '.join(expected_params) + ')->' + rt
        if program[0] == "deref":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            ot2 = outp_type(program[2], args)
            if ot2 != "int":
                return None
            return ot1[:-2]
        if program[0] == "is_sorted":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            return "bool"
        if program[0] == "len":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            return "int"
        if program[0] == "head":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            return ot1[:-2]
        if program[0] == "sort":
            ot1 = outp_type(program[1], args)
            if not is_array_type(ot1):
                return None
            return ot1
        return None
    else:
        if program in args:
            return args[program]
        if program in literal_map:
            ret = literal_map[program]
            if ret == "comparison":
                if len(expected_params) != 2:
                    return None
                if expected_params[0] != expected_params[1]:
                    return None
                return '(' + ', '.join(expected_params) + ')->bool'
            return ret
        return None

def valid_program_for_args(program, args):
    return outp_type(program, args) is not None

