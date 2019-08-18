#include <iostream>
#include <algorithm>
#include "../CNN/cnn.h"
#include "../CNN/utils.h"

using namespace std;

int main() {
    const char* dataPath = "train-images.idx3-ubyte";
    const char* labelPath = "train-labels.idx1-ubyte";

    // load training data
    vector<case_t> cases = read_test_cases(dataPath, labelPath);

    vector<layer_t *> layers;

    layers.emplace_back((layer_t *) new conv_layer_t(1, 5, 8, cases[0].data.size)); // 28 * 28 * 1 -> 24 * 24 * 8
    layers.emplace_back((layer_t *) new relu_layer_t(layers.back()->out.size));
    layers.emplace_back((layer_t *) new pool_layer_t(2, 2, layers.back()->out.size)); // 24 * 24 * 8 -> 12 * 12 * 8
    layers.emplace_back((layer_t *) new fc_layer_t(layers.back()->out.size, 10)); // 4 * 4 * 16 -> 10

    float amse = 0;
    int ic = 0;

    for (long ep = 0; ep < 100000;) {

        for (case_t &t : cases) {
            float xerr = train(layers, t.data, t.out);
            amse += xerr;

            ep++;
            ic++;

            if (ep % 1000 == 0)
                cout << "case " << ep << " err=" << amse / (float) ic << endl;

            // if ( GetAsyncKeyState( VK_F1 ) & 0x8000 )
            // {
            //	   printf( "err=%.4f%\n", amse / ic  );
            //	   goto end;
            // }
        }
    }
    // end:

    const char* testDataPath = "t10k-images.idx3-ubyte";
    const char* testLabelPath = "t10k-labels.idx1-ubyte";

    vector<case_t> testCases = read_test_cases(testDataPath, testLabelPath);

    int correct = 0;
    for (case_t &t : testCases) {
        forward(layers, t.data);
        tensor_t<float> &out = layers.back()->out;

        // find max
        int maxIndex = 0;
        float maxElement = 0;
        int labelIndex = 0;
        for (int i = 0; i < 10; i++) {
            if (out(i, 0, 0) > maxElement) {
                maxElement = out(i, 0, 0);
                maxIndex = i;
            }
            if (t.out(i, 0, 0) == 1) {
                labelIndex = i;
            }
        }
        if (maxIndex == labelIndex)
            correct++;
    }

    cout << "correct: " << (float) correct / (float) testCases.size() * 100.0f << "%" << endl;
    return 0;
}
