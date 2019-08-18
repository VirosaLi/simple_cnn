//
// Created by fangchenli on 8/15/19.
//

#ifndef SIMPLE_CNN_UTILS_H
#define SIMPLE_CNN_UTILS_H

#include "cnn.h"
#include "byteswap.h"

#include <fstream>

struct case_t {
    tensor_t<float> data;
    tensor_t<float> out;
};

uint8_t *read_file(const char *szFile) {
    std::ifstream file(szFile, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size == -1)
        return nullptr;

    auto *buffer = new uint8_t[size];
    file.read((char *) buffer, size);
    return buffer;
}

std::vector<case_t> read_test_cases(const char* dataPath, const char * labelPath) {
    std::vector<case_t> cases;

    uint8_t *train_image = read_file(dataPath);
    uint8_t *train_labels = read_file(labelPath);

    uint32_t case_count = byteswap_uint32(*(uint32_t *) (train_image + 4));

    for (size_t i = 0; i < case_count; i++) {
        case_t c
                {
                        tensor_t<float>(28, 28, 1),
                        tensor_t<float>(10, 1, 1)
                };

        uint8_t *img = train_image + 16 + i * (28 * 28);
        uint8_t *label = train_labels + 8 + i;

        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++)
                c.data(x, y, 0) = (float) img[x + y * 28] / 255.f;

        for (int b = 0; b < 10; b++)
            c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;

        cases.push_back(c);
    }
    delete[] train_image;
    delete[] train_labels;

    return cases;
}

void forward(std::vector<layer_t *> &layers, tensor_t<float> &data) {
    for (size_t i = 0; i < layers.size(); i++) {
        if (i == 0)
            activate(layers[i], data);
        else
            activate(layers[i], layers[i - 1]->out);
    }
}

float train(std::vector<layer_t *> &layers, tensor_t<float> &data, tensor_t<float> &expected) {
    for (size_t i = 0; i < layers.size(); i++) {
        if (i == 0)
            activate(layers[i], data);
        else
            activate(layers[i], layers[i - 1]->out);
    }

    tensor_t<float> grads = layers.back()->out - expected;

    for (int i = layers.size() - 1; i >= 0; i--) {
        if (i == layers.size() - 1)
            calc_grads(layers[i], grads);
        else
            calc_grads(layers[i], layers[i + 1]->grads_in);
    }

    for (auto &layer : layers) {
        fix_weights(layer);
    }

    float err = 0;
    for (int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++) {
        float f = expected.data[i];
        if (f > 0.5)
            err += std::abs(grads.data[i]);
    }
    return err * 100;
}

int save_network(const std::vector<layer_t *> &layers, const char *path) {
    std::ofstream out(path);
    for (auto &layer : layers) {
        out << toString(layer);
    }
    out.close();
    return 0;
}

std::vector<layer_t *> load_network(const char *path) {
    std::vector<layer_t *> layers_loaded;
    std::ifstream infile;
    infile.open(path, std::ios::in);
    if (infile.is_open()) {

        std::string line;
        while (getline(infile, line)) {

            if (line.length() != 0) {

                if (line == "fc") {
                    getline(infile, line);
                    tensor_t<float> tensor_in = string_to_tensor(line);

                    getline(infile, line);
                    tensor_t<float> tensor_out = string_to_tensor(line);

                    getline(infile, line);
                    tensor_t<float> tensor_weight = string_to_tensor_trans(line);

                    getline(infile, line);
                    tensor_t<float> tensor_grads_in = string_to_tensor(line);

                    layers_loaded.emplace_back(
                            (layer_t *) new fc_layer_t(tensor_in, tensor_out, tensor_weight, tensor_grads_in));
                }

                if (line == "conv") {
                    getline(infile, line);
                    tensor_t<float> tensor_in = string_to_tensor(line);

                    getline(infile, line);
                    tensor_t<float> tensor_out = string_to_tensor(line);

                    getline(infile, line);
                    tensor_t<float> tensor_grads_in = string_to_tensor(line);

                    std::vector<tensor_t<float>> filters;
                    while (true) {
                        getline(infile, line);
                        if (line == "end") {
                            break;
                        }
                        filters.push_back(string_to_tensor(line));
                    }

                    getline(infile, line);
                    int extend_filter = stoi(line);

                    getline(infile, line);
                    int stride = stoi(line);

                    layers_loaded.emplace_back(
                            (layer_t *) new conv_layer_t(tensor_in, tensor_out, tensor_grads_in, filters, extend_filter,
                                                         stride));
                }

                if (line == "relu") {
                    getline(infile, line);
                    tensor_t<float> tensor_in = string_to_tensor(line);

                    getline(infile, line);
                    tensor_t<float> tensor_out = string_to_tensor(line);

                    getline(infile, line);
                    tensor_t<float> tensor_grads_in = string_to_tensor(line);

                    layers_loaded.emplace_back((layer_t *) new relu_layer_t(tensor_in, tensor_out, tensor_grads_in));
                }

                if (line == "pool") {
                    getline(infile, line);
                    tensor_t<float> tensor_in = string_to_tensor(line);

                    getline(infile, line);
                    tensor_t<float> tensor_out = string_to_tensor(line);

                    getline(infile, line);
                    tensor_t<float> tensor_grads_in = string_to_tensor(line);

                    getline(infile, line);
                    int extend_filter = stoi(line);

                    getline(infile, line);
                    int stride = stoi(line);

                    layers_loaded.emplace_back(
                            (layer_t *) new pool_layer_t(tensor_in, tensor_out, tensor_grads_in, extend_filter,
                                                         stride));
                }
            }
        }
        infile.close();
    }
    return layers_loaded;
}

void test(std::vector<layer_t *> &layers, const char *path) {
    uint8_t *data = read_file(path);
    if (data) {
        uint8_t *usable = data;

        while (*(uint32_t *) usable != 0x0A353532)
            usable++;

#pragma pack(push, 1)
        struct RGB {
            uint8_t r, g, b;
        };
#pragma pack(pop)

        RGB *rgb = (RGB *) usable;

        tensor_t<float> image(28, 28, 1);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                RGB rgb_ij = rgb[i * 28 + j];
                image(j, i, 0) = ((((float) rgb_ij.r
                                    + (float) rgb_ij.g
                                    + (float) rgb_ij.b)
                                   / (3.0f * 255.f)));
            }
        }

        std::cout << "inference result: " << std::endl;
        forward(layers, image);
        tensor_t<float> &out = layers.back()->out;
        for (int i = 0; i < 10; i++) {
            printf("[%i] %f\n", i, out(i, 0, 0) * 100.0f);
        }
    }
}

#endif //SIMPLE_CNN_UTILS_H
