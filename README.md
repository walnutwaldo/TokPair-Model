The TokPair Model
=====
This project was created with MIT PRIMES. The work is described in the [paper](https://math.mit.edu/research/highschool/primes/materials/2020/Yan-Moses.pdf) "Token Pairing to Improve Neural Program Synthesis Models".

**Abstract**

In neural program synthesis (NPS), a network is trained to output or aid in theoutput of code that satisfies a given program specification. In our work, we make modifications upon the simple sequence-to-sequence (Seq2Seq) LSTM model. Extending the most successful techniques from previous works, we guide a beam search with an encoder-decoder scheme augmented with attention mechanismsand a specialized syntax layer. But one of the withstanding difficulties of NPS isthe implicit tree structure of programs, which makes it inherently more difficult for linearly-structured models. To address this, we experiment with a novel technique we call token pairing. Our model is trained and evaluated on AlgoLisp, a dataset of English description-to-code programming problems paired with example solutions and test cases on which to evaluate programs. We also create a new interpreter for AlgoLisp that fixes the bugs present in the builtin executor. In the end, our model achieves 99.24% accuracy at evaluation, which greatly improveson the previous state-of-the-art of 95.80% while using fewer parameters.

# Using the Code
This is a quick guide to start using the codebase. These instructions will run you through training and evaluating a TokPair-150 model on a cleaned dataset as described in the [paper](https://math.mit.edu/research/highschool/primes/materials/2020/Yan-Moses.pdf).
## Prerequisites and Installation
Prerequisites and installation for using the AlgoLisp dataset and NearAI's tools are decribed [here](https://github.com/nearai/program_synthesis) as follows:
### Prerequisites
Python 3 (>=3.5) is required to run the code. We also recommend using  [virtualenv](https://virtualenv.pypa.io/en/stable/)  for isolated Python environments and  [pip](https://pypi.org/project/pip/)  for package management. Note, to create a Python 3 environment you need to run:
```
virtualenv .env --python=python3
source .env/bin/activate
```
The code also assumes that  [PyTorch](https://pytorch.org/)  is already installed.
### Installation

Clone and install the repository https://github.com/nearai/program_synthesis (you may download and install it directly within the root directory).

```
git clone https://github.com/nearai/program_synthesis.git
cd program_synthesis
```

Install program-synthesis in editable mode:

```
pip install -e .
```

## Downloading the Datasets

First, download the datasets (train, dev, and eval) from https://github.com/nearai/program_synthesis/tree/master/program_synthesis/algolisp.
and unpack the three files into the `data` folder.

### Cleaning and Processing Data
To filter out invalid data, run `filter.py` within the `data/` folder:
```
cd data
python filter.py
```
The filtered data will be saved in the `filtered_data2/` folder. In this new folder, generate the vocabulary and sketches files:
```
cd ../filtered_data2
python vocabgen.py
python sketchgen.py
```
Now we will encode the data with 150 new tokens created from token pairing and divide the dataset into 4 curriculums (the test dataset is also divided despite not being used during training):
```
python encode.py 150 4
```
These new datasets are found in the `filtered_data2/encoded` directory. Within the directory is a tool, `analyze.py`, which gives some statistics on the datsets and should report an assertion error if the data has any issues. You can run it as follows:
```
cd encoded
python analyze.py 150 4
```
### Training
The files are configured for TokPair-150 using 4 curriculums. This may be changed. From the root directory, you may edit `src/datasets.py`. The global variables `data_group`, `num_tokens`, `vocab_size`, `inp_size`, and `outp_size` all must be changed and comments in the file describe how to find the proper values for a desired training configuration.

Next, to train a model, run the file `src/train.py` from the root directory. There are several flags you may use to modify the training procedure and specify the location for saving the model. Here is an example:
```
python src/train.py --save_dir=saved_models/my_first_model/model --num_epochs=10 --num_curriculums=4 --batch_size=32
```
### Evaluating
Once a model is trained, find the latest saved version. In case you used the provided training command, the latest saved version should look like `saved_models/my_first_model/model-NUMBER`. To evaluate the model on the test data, run:
```
python src/eval.py --beam_size=10 --save_dir=saved_models/my_first_model/model-3 --dataset=test
```
Changing the dataset flag to `dev` or `train` will evaluate the models on the dev and training datsets resepectively. The beam size among other variables can be changed via flags as well.
