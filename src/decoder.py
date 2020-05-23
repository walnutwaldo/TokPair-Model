import datasets
import json

trees = {}
loaded_sketches = False

def load_sketches():
    with open('../' + ('filtered_' if datasets.filtered else '') + 'data/sketches.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            trees[data['id']] = data['tree']
            if len(trees) == datasets.vocab_size:
                break

    loaded_sketches = True

def count_holes(tree):
    return sum([count(x) if type(x) is list else (1 if x == "<HOLE>" else 0) for x in tree])

def replace_holes(tree, replacements):
    res = [replace_holes(x, replacements) if type(x) is list else (replacements.pop(0) if x == "<HOLE>" else x) for x in tree]
    if len(res) == 1:
        return res[0]
    return res

def build_tree(consumable):
    if not consumable:
        return []
    root = consumable.pop(0)
    tree = trees[root]
    holes = count_holes(tree)
    replacements = [build_tree(consumable) for _ in range(holes)]

    for x in replacements:
        if not x:
            return []
    return replace_holes(tree, replacements)

def convert_to_short_tree(linear_encoded_program):
    if not loaded_sketches:
        load_sketches()
    consumable = linear_encoded_program[:]
    res = build_tree(consumable)
    if consumable:
        return []
    return res

