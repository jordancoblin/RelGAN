## RelGAN

The core code in this repo was forked from https://github.com/weilinie/RelGAN. This repo adds modifications for sparse alternatives of the softmax activation function during sampling.

## Dependencies
This project uses Python 3.5.2, with the following lib dependencies:
* [Tensorflow 1.4](https://www.tensorflow.org/)
* [Numpy 1.14.1](http://www.numpy.org/)
* [Matplotlib 2.2.0](https://matplotlib.org)
* [Scipy 1.0.0](https://www.scipy.org)
* [NLTK 3.2.3](https://www.nltk.org)
* [tqdm 4.19.6](https://pypi.python.org/pypi/tqdm)

## Installation Instructions

Because the original repo uses an old version of TF, we also need a compatible version of Python. For this purpose, we will use `pyenv`.

1. Follow instructions for installing pyenv: https://github.com/pyenv/pyenv#getting-pyenv
2. Follow instructions for installing pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv
3. Install python version 3.6.15 using pyenv (this was the only one whose build succeeded for me): `pyenv install 3.6.15`
4. Create a new virtual environment using pyenv-virtualenv: `pyenv virtualenv 3.6.15 relgan-3.6.15`
5. Activate the virtual env: `pyenv activate relgan-3.6.15`
6. (Finally) install tensorflow: `pip install tensorflow-cpu==1.15.0`
  You can check that the installation works by trying out some [tf operations](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/g3doc/get_started/os_setup.md#run-tensorflow-from-the-command-line).
8. And install the rest of the dependencies as well: `pip install numpy matplotlib scipy nltk tqdm`
9. Profit $$$


## Instructions
The `experiments` folders contain scripts for starting the different experiments.
For example, to reproduce the `synthetic data` experiments, you can try:
```
cd oracle/experiments
python3 oracle_relgan.py [job_id] [gpu_id]
```
or `COCO Image Captions`:
```
cd real/experiments
python3 coco_relgan.py [job_id] [gpu_id]
```
or `EMNLP2017 WMT News`:
```
cd real/experiments
python3 emnlp_relgan.py [job_id] [gpu_id]
```
Note to replace [job_id] and [gpu_id] with appropriate numerical values.

## Reference
To cite this work, please use
```
@INPROCEEDINGS{Nie2019ICLR,
  author = {Nie, Weili and Narodytska, Nina and Patel, Ankit},
  title = {RelGAN: Relational Generative Adversarial Networks for Text Generation},
  booktitle = {International conference on learning representations (ICLR)},
  year = {2019}
}
```

## Acknowledgement
This code is based on the previous benchmarking platform [Texygen](https://github.com/geek-ai/Texygen). 
