import json
import sys
from collections import Counter

vocab_size = 100
word_counter = Counter()

def fill_token_set(token_set, tree):
    if type(tree) is list:
        for a in tree:
            fill_token_set(token_set, a)
    else:
        token_set.add(tree)

def analyze(file_name):
    global word_counter
    count = [0] * vocab_size
    with open(file_name, 'r') as f:
        for line in f:
            problem = json.loads(line)
            encoded_tree = problem['encoded_tree']
            token_set = set()
            fill_token_set(token_set, encoded_tree)
            for token in token_set:
                count[token] += 1
            text = problem['text']
            word_counter.update(text)
    return count

def main():
    global vocab_size
    if len(sys.argv) > 1:
        vocab_size = int(sys.argv[1])
    counts = {}
    for d_set in 'test dev train'.split():
        counts[d_set] = analyze(d_set + '-' + str(vocab_size) + '.jsonl')
    print(sorted([(counts['train'][i], counts['dev'][i], counts['test'][i], i) for i in range(vocab_size)]))
    print(word_counter.most_common())

    return 0

if __name__ == '__main__':
    exit(main())
