import json
from program_synthesis.algolisp.dataset import executor
import os
import sys
sys.path.append('../src/')
import runner

ex = executor.LispExecutor()

def passes_evaluation(data):
    for test in data['tests']:
        haderror = False
        try:
            outp = runner.run_program(data['short_tree'], test['input'])
        except:
            haderror = True
    
        if haderror or outp != test['output']:
            runner.debug = True
            print(test['input'])
            print(test['output'])
            print(data['short_tree'])
            runner.run_program(data['short_tree'], test['input'])
            return False
    return True
    #evaluation = executor.evaluate_code(data['short_tree'], data['args'], data['tests'], ex)
    #return evaluation['tests-passed'] == evaluation['tests-executed']

def check_dataset(original_file):
    print('processing %s ... 0' % original_file, end='\r', flush=True)
    filtered_data = []
    lines = []
    with open(original_file, 'r') as f:
        for line in f:
            lines.append(line)
    for i, line in enumerate(lines):
        data = json.loads(line)
        program = data['short_tree']

        only_words = True
        for word in data['text']:
            if ' ' in word:
                only_words = False
                break

        assert (only_words and passes_evaluation(data))
        print('processing %s ... %d' % (original_file, i + 1), end='\r', flush=True)
            
    print('processing %s ... DONE   ' % original_file, flush=True)

def main():
    for t in 'train dev test'.split():
        check_dataset('metaset3.' + t + '.jsonl')

if __name__ == '__main__':
    exit(main())
