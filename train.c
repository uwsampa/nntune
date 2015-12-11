#include "fann.h"
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

const unsigned int MAX_LAYERS = 16;
const float DESIRED_ERROR = 0.0001;

/*
Helper function to determine if a file exists
*/
int file_exists(const char* fname) {
    return (access(fname, F_OK)!=-1);
}
/*
Helper function to remove file extension
*/
void trim_ext(const char* mystr, char* retstr) {
    char *lastdot;
    strcpy (retstr, mystr);
    lastdot = strrchr (retstr, '.');
    if (lastdot != NULL)
        *lastdot = '\0';
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
    // String buffer
    char fn[256];
    char intbuf[8];

    // Argument validation
    if (argc != 7 && argc != 8) {
        printf("Error: Incorrect number of arguments!\n");
        printf("  usage:   ./train <training dataset> <topology> <epochs> <learning rate> <fixed precision> <nn file> <optional: test dataset>\n");
        printf("  example: ./train train.data 18-4-2 100 0.2 8 output\n");
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

    // Argument 5: decimal precision
    unsigned int precision = atoi(argv[5]);
    sprintf(intbuf, "%d", precision);

    // Argument 6: output filename.
    const char *outfn = argv[6];

    // Argument 7: test data filename.
    const char *testdatafn = "";
    if (argc==8) {
        testdatafn = argv[7];
        assert(file_exists(testdatafn));
    }

    // ANN
    struct fann *ann;
    ann = fann_create_standard_array(num_layers, layer_sizes);

    // Training data
    struct fann_train_data *data;
    data = fann_read_train_from_file(datafn);
    if (precision!=0) {
        unsigned i, j;
        unsigned count = fann_length_train_data(data);
        unsigned num_outputs = fann_num_output_train_data(data);
        unsigned num_inputs = fann_num_input_train_data(data);
        int temp;
        for(i = 0; i < count; ++i)
        {
            for (j = 0; j < num_inputs; ++j) {
                temp = (int) ((data->input[i][j]*pow(2,precision))+0.5);
                data->input[i][j] = ((float) temp) / pow(2,precision);
            }
            for (j = 0; j < num_outputs; ++j) {
                temp = (int) ((data->output[i][j]*pow(2,precision))+0.5);
                data->output[i][j] = ((float) temp) / pow(2,precision);
            }
        }
    }

    // Misc parameters.
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    fann_set_activation_steepness_hidden(ann, 1);
    fann_set_activation_steepness_output(ann, 1);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    // Set the learning rate (probably will get ignored)
    fann_set_learning_rate(ann, learning_rate);

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

    if (precision) {

        // // Determine the decimal precision requirements
        // trim_ext(outfn, fn);
        // strcat(fn, "_fix");
        // strcat(fn, ".nn");
        // unsigned decimal = fann_save_to_fixed(ann, fn);
        // printf("Fixed point configuration has %d decimal points\n", decimal);

        // int i;
        // for (i=precision; i<decimal; i++) {
        //     sprintf(intbuf, "%d", i);

        //     // Dump the ANN specification (fixed)
        //     trim_ext(outfn, fn);
        //     strcat(fn, "_fix_");
        //     strcat(fn, intbuf);
        //     strcat(fn, ".nn");
        //     fann_save_to_fixed_reduced_precision(ann, fn, i);

        //     // Dump the fixed point data
        //     trim_ext(datafn, fn);
        //     strcat(fn, "_fix_");
        //     strcat(fn, intbuf);
        //     strcat(fn, ".data");
        //     fann_save_train_to_fixed(data, fn, i);
        // }

        fann_save_to_fixed_reduced_precision(ann, outfn, precision);
        fann_save_train_to_fixed(data, datafn, precision);

        // Training data
        if (strcmp (testdatafn,"") != 0) {
            struct fann_train_data *test_data;
            test_data = fann_read_train_from_file(testdatafn);
            fann_save_train_to_fixed(test_data, testdatafn, precision);
        }

    } else {
        // Dump the ANN specification
        fann_save(ann, outfn);
    }

    fann_destroy(ann);
    fann_destroy_train(data);

    return 0;
}
