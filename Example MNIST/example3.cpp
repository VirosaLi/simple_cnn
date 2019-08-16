//
// Created by fangchenli on 8/13/19.
//

#include <iostream>
#include <algorithm>
#include "../CNN/cnn.h"
#include "../CNN/utils.h"

using namespace std;

int main() {

    // load training data
    vector<case_t> cases = read_test_cases();

    // build network
    vector<layer_t *> layers;
    layers.emplace_back((layer_t *) new conv_layer_t(1, 5, 8, cases[0].data.size)); // 28 * 28 * 1 -> 24 * 24 * 8
    layers.emplace_back((layer_t *) new relu_layer_t(layers.back()->out.size));
    layers.emplace_back((layer_t *) new pool_layer_t(2, 2, layers.back()->out.size)); // 24 * 24 * 8 -> 12 * 12 * 8
    layers.emplace_back((layer_t *) new conv_layer_t(1, 3, 10, layers.back()->out.size)); // 12 * 12 * 6 -> 10 * 10 * 10
    layers.emplace_back((layer_t *) new relu_layer_t(layers.back()->out.size));
    layers.emplace_back((layer_t *) new pool_layer_t(2, 2, layers.back()->out.size)); // 10 * 10 * 10 -> 5 * 5 * 10
    layers.emplace_back((layer_t *) new fc_layer_t(layers.back()->out.size, 10)); // 4 * 4 * 16 -> 10

    // train
    float amse = 0;
    int ic = 0;
    for (long ep = 0; ep < 1000;) {

        for (case_t &t : cases) {
            float xerr = train(layers, t.data, t.out);
            amse += xerr;

            ep++;
            ic++;

            if (ep % 1000 == 0)
                cout << "case " << ep << " err=" << amse / (float) ic << endl;

            // if ( GetAsyncKeyState( VK_F1 ) & 0x8000 )
            // {
            //     printf( "err=%.4f%\n", amse / ic  );
            //     goto end;
            // }
        }
    }

    const char *path = "nn_model";

    // save model
    save_network(layers, path);

    // load model
    vector<layer_t *> layers_loaded = load_network(path);

    // test save and load
    const char *testPath = "test.ppm";
    test(layers, testPath);
    test(layers_loaded, testPath);
}
