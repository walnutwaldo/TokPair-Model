import json
import sys
from collections import Counter

vocab_size = 100
all_words = set()

def fill_token_set(token_set, tree):
    if type(tree) is list:
        for a in tree:
            fill_token_set(token_set, a)
    else:
        token_set.add(tree)

def analyze(file_name):
    count = [0] * vocab_size
    word_counter = Counter()
    with open(file_name, 'r') as f:
        for line in f:
            problem = json.loads(line)
            encoded_tree = problem['encoded_tree']
            token_set = set()
            fill_token_set(token_set, encoded_tree)
            for token in token_set:
                count[token] += 1
            text = problem['text']
            all_words.update(text)
            word_counter.update(text)
    return count, word_counter

def main():
    global vocab_size
    if len(sys.argv) > 1:
        vocab_size = int(sys.argv[1])
    token_counts, word_counts = {}, {}
    for d_set in 'test dev train'.split():
        token_counts[d_set], word_counts[d_set] = analyze(d_set + '-' + str(vocab_size) + '.jsonl')
    for i in range(vocab_size):
        assert(token_counts['train'][i] > 0)
        assert(token_counts['train'][i] > 0)
    for k in all_words:
        assert(word_counts['train'][k] > 0)
        assert(word_counts['train'][k] > 0)
    print(sorted([(token_counts['train'][i], token_counts['dev'][i], token_counts['test'][i], i) for i in range(vocab_size)]))
    print(sorted([(word_counts['train'][k], word_counts['dev'][k], word_counts['test'][k], k) for k in all_words]))
    print("%d words"%len(all_words))

    return 0

if __name__ == '__main__':
    exit(main())
