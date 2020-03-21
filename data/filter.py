import json
from program_synthesis.algolisp.dataset import executor
import os

def filter_dataset(original_file, new_file):
    ex = executor.LispExecutor()
    print('processing %s ... 0' % original_file, end='\r', flush=True)
    line_cnt = 0
    filtered_data = []
    with open(original_file, 'r') as f:
        for line in f:
            line_cnt += 1
            data = json.loads(line)
            program = data['short_tree']
            evaluation = executor.evaluate_code(data['short_tree'], data['args'], data['tests'], ex)
            if evaluation['tests-passed'] == evaluation['tests-executed']:
                filtered_data.append(data)

            print('processing %s ... %d' % (original_file, line_cnt), end='\r', flush=True)
            
    print('processing %s ... DONE   ' % original_file, flush=True)
    print('saving %s ... 0' % new_file, end='\r', flush=True)
    with open(new_file, 'w') as f:
        for i, data in enumerate(filtered_data):
            json.dump(data, f)
            f.write('\n')
            if i % 100 == 0:
                print('saving %s ... %d' % (new_file, i + 1), end='\r', flush=True)
    print('saving %s ... DONE     ' % new_file, flush=True)

def main():
    if not os.path.exists('../filtered_data/'):
        os.mkdir('../filtered_data/')
    for t in 'train dev test'.split():
        filter_dataset('metaset3.' + t + '.jsonl', '../filtered_data/meatset3.' + t + '.jsonl')

if __name__ == '__main__':
    exit(main())
