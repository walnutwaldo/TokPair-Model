import json
import tensorflow as tf

dataset = 'filtered_data'
vocab_size = 81

def import_dataset(dset_type):
    problems = []
    with open('%s/encoded/%s-%d.jsonl'%(dataset, dset_type, vocab_size), 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems.append(problem)
    texts = [problem['text'] for problem in problems]
    programs = [problem['encoded_tree'] for problem in problems]
    tests = [problem['tests'] for problem in problems]
    return problems

train_dataset = import_dataset('train')
dev_dataset = import_dataset('dev')
eval_dataset = import_dataset('test')

def main():
    pass

if __name__ == '__main__':
    exit(main())
