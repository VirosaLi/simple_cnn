cmake_minimum_required(VERSION 3.14)
project(simple_cnn)

set(CMAKE_CXX_STANDARD 14)

include_directories(CNN)
include_directories("Example MNIST")

set(CNN_CORE
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
        CNN/utils.h)

add_executable(example1
        ${CNN_CORE}
        "Example MNIST/example1.cpp")

add_executable(example2
        ${CNN_CORE}
        "Example MNIST/example2.cpp")

add_executable(test
        ${CNN_CORE}
        "Example MNIST/example3.cpp")
