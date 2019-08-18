# simple_cnn

## Forked Repo for CSE 569S
This repo is forked and refactored from https://github.com/can1357/simple_cnn for CSE 569S.

Changes made in the framework include:

* Added MNIST test dataset.
* Added save and load functions.
* Removed the real-time inference from the example. See the original repo if you are interested.
* Replaced deprecated libraries. 
* Reformatted code.
* And more.

If you don't like this this version, you are welcome to directly use the original program from the link above.


## Build

Make sure cmake is installed.

First generate the Makefile using cmake:

    cmake .

Then build the program:

    make


MNIST digits taken from http://yann.lecun.com/exdb/mnist/
