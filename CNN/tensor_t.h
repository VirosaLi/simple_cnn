#pragma once

#include "point_t.h"
#include <cassert>
#include <vector>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <iterator>

template<typename T>
struct tensor_t {
    T *data;

    tdsize size{};

    tensor_t(int _x, int _y, int _z) {
        data = new T[_x * _y * _z];
        size.x = _x;
        size.y = _y;
        size.z = _z;
    }

    tensor_t(const tensor_t &other) {
        data = new T[other.size.x * other.size.y * other.size.z];
        memcpy(
                this->data,
                other.data,
                other.size.x * other.size.y * other.size.z * sizeof(T)
        );
        this->size = other.size;
    }

    tensor_t<T> operator+(tensor_t<T> &other) {
        tensor_t<T> clone(*this);
        for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++)
            clone.data[i] += other.data[i];
        return clone;
    }

    tensor_t<T> operator-(tensor_t<T> &other) {
        tensor_t<T> clone(*this);
        for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++)
            clone.data[i] -= other.data[i];
        return clone;
    }

    T &operator()(int _x, int _y, int _z) {
        return this->get(_x, _y, _z);
    }

    T &get(int _x, int _y, int _z) {
        assert(_x >= 0 && _y >= 0 && _z >= 0);
        assert(_x < size.x && _y < size.y && _z < size.z);

        return data[
                _z * (size.x * size.y) +
                _y * (size.x) +
                _x
        ];
    }

    void copy_from(std::vector<std::vector<std::vector<T>>> vector) {
        int z = vector.size();
        int y = vector[0].size();
        int x = vector[0][0].size();

        for (int i = 0; i < x; i++)
            for (int j = 0; j < y; j++)
                for (int k = 0; k < z; k++)
                    get(i, j, k) = vector[k][j][i];
    }

    ~tensor_t() {
        delete[] data;
    }
};

static void print_tensor(tensor_t<float> &data) {
    int mx = data.size.x;
    int my = data.size.y;
    int mz = data.size.z;

    for (int z = 0; z < mz; z++) {
        printf("[Dim%d]\n", z);
        for (int y = 0; y < my; y++) {
            for (int x = 0; x < mx; x++) {
                printf("%.2f \t", (float) data.get(x, y, z));
            }
            printf("\n");
        }
    }
}

static tensor_t<float> to_tensor(std::vector<std::vector<std::vector<float>>> data) {
    int z = data.size();
    int y = data[0].size();
    int x = data[0][0].size();


    tensor_t<float> t(x, y, z);

    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            for (int k = 0; k < z; k++)
                t(i, j, k) = data[k][j][i];
    return t;
}

static std::string tensor_to_string(tensor_t<float> &data) {
    std::stringstream ss;
    ss << data.size.x << " " << data.size.y << " " << data.size.z << " ";

    for (int z = 0; z < data.size.z; ++z) {
        for (int y = 0; y < data.size.y; ++y) {
            for (int x = 0; x < data.size.x; ++x) {
                ss << data.get(x, y, z) << " ";
            }
        }
    }
    return ss.str();
}

static tensor_t<float> string_to_tensor(const std::string& s) {
    std::istringstream iss(s);
    std::vector<float> v((std::istream_iterator<float>(iss)), std::istream_iterator<float>());
    int x = (int) v[0];
    int y = (int) v[1];
    int z = (int) v[2];/**/

    tensor_t<float> t(x, y, z);
    for (int i = 0; i < z; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < x; ++k) {
                t(k, j, i) = v[ x*y*i + y*j + k + 3];
            }
        }
    }
    return t;
}