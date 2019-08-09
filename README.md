# simple_cnn

## Forked Repo for CSE 569S
This repo is forked and refactored from https://github.com/can1357/simple_cnn for CSE 569S.

## Original Description from the author

simple_cnn is ment to be an easy to read and easy to use convolutional neural network library.

simple_cnn is written in a mostly C-like manner behind the scenes, doesnt use virtual classes and avoids using std where its possible so that it is easier to convert to CUDA code when needed.


Example use on handwritten digit recognition (Youtube Video):

[![Youtube Video](https://img.youtube.com/vi/afLUb6lFTCk/0.jpg)](https://www.youtube.com/watch?v=afLUb6lFTCk)

# Building

Make sure cmake is installed.

First generate the Makefile using cmake:

    cmake .

Then build the program:

    make


MNIST digits taken from http://yann.lecun.com/exdb/mnist/
