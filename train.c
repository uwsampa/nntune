#include "fann.h"
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

const unsigned int MAX_LAYERS = 16;
const float DESIRED_ERROR = 0.01;

/*
Helper function to determine if a file exists
*/
int file_exists(const char* fname) {
    return (access(fname, F_OK)!=-1);
}

/*
arguments (all required):
 - training data filename
 - topology, as number of neurons per layer separated by dashes
 - epochs (integer)
 - learning rate (0.0-1.0 float)
 - output filename
*/
int main(int argc, char **argv)
{
    // Argument validation
    if (argc != 6) {
        printf("Error: Incorrect number of arguments!\n");
        printf("  usage:   ./train <training dataset> <topology> <epochs> <learning rate> <nn file>\n");
        printf("  example: ./train train.data 18-4-2 100 0.2 output.nn\n");
        return 0;
    }

    // Argument 1: training data filename.
    const char *datafn = argv[1];
    assert(file_exists(datafn));

    // Argument 2: topology.
    unsigned int layer_sizes[MAX_LAYERS];
    unsigned int num_layers = 0;
    char *token = strtok(argv[2], "-");
    while (token != NULL) {
        assert(atoi(token)!=0);
        layer_sizes[num_layers] = atoi(token);
        ++num_layers;
        token = strtok(NULL, "-");
    }

    // Argument 3: epoch count.
    unsigned int max_epochs = atoi(argv[3]);
    assert(max_epochs>50);

    // Argument 4: learning rate.
    float learning_rate = atof(argv[4]);
    assert(learning_rate>0&&learning_rate<1);

    // Argument 5: output filename.
    const char *outfn = argv[5];

    // ANN
    struct fann *ann;
    ann = fann_create_standard_array(num_layers, layer_sizes);

    // Misc parameters.
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    fann_set_activation_steepness_hidden(ann, 1);
    fann_set_activation_steepness_output(ann, 1);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    // Set the learning rate (probably will get ignored)
    fann_set_learning_rate(ann, learning_rate);

    // Training data
    struct fann_train_data *data;
    data = fann_read_train_from_file(datafn);

    // Initialize training weights based on the training data
    fann_init_weights(ann, data);

    // Start the training!
    fann_train_on_data(
        ann,
        data,
        max_epochs,
        1000,  // epochs between reports
        DESIRED_ERROR
    );

    // Evaluating the training data
    printf("Testing network on training data. MSE=%f\n", fann_test_data(ann, data));

    // Dump the ANN specification
    fann_save(ann, outfn);

    fann_destroy_train(data);
    fann_destroy(ann);

    return 0;
}
