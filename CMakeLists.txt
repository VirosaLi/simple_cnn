cmake_minimum_required(VERSION 3.14)
project(simple_cnn)

set(CMAKE_CXX_STANDARD 14)

include_directories(CNN)
include_directories("Example MNIST")

add_executable(simple_cnn
        CNN/cnn.h
        CNN/conv_layer_t.h
        CNN/dropout_layer_t.h
        CNN/fc_layer.h
        CNN/gradient_t.h
        CNN/layer_t.h
        CNN/optimization_method.h
        CNN/point_t.h
        CNN/pool_layer_t.h
        CNN/relu_layer_t.h
        CNN/tensor_t.h
        CNN/types.h
        "Example MNIST/byteswap.h"
        "Example MNIST/example1.cpp"
        #"Example MNIST/example2.cpp"
        "Example MNIST/sleep.h")
