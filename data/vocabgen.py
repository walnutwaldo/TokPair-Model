from collections import Counter
import json

words = Counter()

def expand(text):
    res = []
    for t in text:
        w = t.split()
        for word in w:
            curr = ''
            for c in word:
                if c >= 'a' and c <= 'z':
                    curr += c
                else:
                    if curr:
                        res.append(curr)
                        curr = ''
                    res.append(c)
            if curr:
                res.append(curr)
    return res

def add_words(file_name):
    global words
    lines = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            lines.append(line)
    for i, line in enumerate(lines):
        problem = json.loads(line)
        text = problem['text']
        text = expand(text)
        words.update(text)
        if (i + 1) % 100 == 0:
            print('processing %s ... %d/%d' % (file_name, i + 1, len(lines)), end='\r', flush=True)
    print('processing %s ... DONE')

def main():
    for d_set in 'train dev test'.split():
        add_words('metaset3.%s.jsonl'%d_set)
    with open('vocab.txt', 'w') as f:
        for i, (w, _) in enumerate(words.most_common()):
            f.write('%d %s\n'%(i, w))

if __name__ == '__main__':
    exit(main())
