import json

vocab_size = 100

def fill_token_set(token_set, tree):
    if type(tree) is list:
        for a in tree:
            fill_token_set(token_set, a)
    else:
        token_set.add(tree)

def analyze(file_name):
    count = [0] * vocab_size
    with open(file_name, 'r') as f:
        for line in f:
            problem = json.loads(line)
            encoded_tree = problem['encoded_tree']
            token_set = set()
            fill_token_set(token_set, encoded_tree)
            for token in token_set:
                count[token] += 1
    return count

def main():
    counts = {}
    for d_set in 'test dev train'.split():
        counts[d_set] = analyze(d_set + '-' + str(vocab_size) + '.jsonl')
    print(sorted([(counts['train'][i], counts['dev'][i], counts['test'][i], i) for i in range(vocab_size)]))

    return 0

if __name__ == '__main__':
    exit(main())
