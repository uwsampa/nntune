/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2012 Steffen Nissen (sn@leenissen.dk)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fann.h"

#define DEBUG NONE
#define ITERATIONS 4
#define printd(lvl, ...) (lvl > DEBUG) ? 0 : printf(__VA_ARGS__)

typedef enum {NONE=0, LITE=1, HIGH=2} debug;

int main()
{
    fann_type *calc_out;
    const unsigned int npu_latency[9][9] = {
        {-1,30,31,33,37,66,132,264,528},
        {-1,37,38,40,44,73,139,271,535},
        {-1,38,39,41,45,74,140,272,536},
        {-1,40,41,43,47,76,142,274,538},
        {-1,47,45,47,51,80,146,278,542},
        {-1,62,62,62,63,99,179,339,659},
        {-1,96,96,96,96,148,260,484,932},
        {-1,164,164,164,164,248,424,776,1480},
        {-1,300,300,300,300,448,752,1360,2576}
    };
    const unsigned int num_neurons[] = {0,1,2,4,8,16,32,64,128};
    const unsigned int epochs[] = {16,32,64,128,256,512,1024};
    const unsigned int num_input = 20;
    const unsigned int num_output = 1;
    const unsigned int num_layers = 4;
    const float desired_error = (const float) 0.0001;
    const unsigned int epochs_between_reports = 0;

    struct fann *ann;
    struct fann_train_data *train_data;
    struct fann_train_data *test_data;

    unsigned int n1 = 0;
    unsigned int n2 = 0;
    unsigned int ep = 0;
    unsigned int it = 0;
    unsigned int confId = 0;

    unsigned int i = 0;
    unsigned int miss = 0;
    float train_miss_rate = 0;
    float test_miss_rate = 0;

    char dstFilePath[256];
    char confIdStr[8];

    FILE * pFile;
    pFile = fopen("results.csv", "w");
    fprintf(pFile, "conf, neurons1, neurons2, epochs, iteration, train miss, train MSE, test miss, test MSE, npu latency\n");

    for(n2 = 0; n2 < sizeof(num_neurons)/sizeof(num_neurons[0]); n2++) {
        for(n1 = 1; n1 < sizeof(num_neurons)/sizeof(num_neurons[0]); n1++) {
            for(ep = 0; ep < sizeof(epochs)/sizeof(epochs[0]); ep++) {
                for(it = 0; it < ITERATIONS; it++) {
                    printd(NONE, "NEURONS %d->%d, \t", num_neurons[n1], num_neurons[n2]);
                    printd(NONE, "EPOCHS %d, \t", epochs[ep]);
                    printd(NONE, "ITERATION %d\n", it);
                    fprintf(pFile, "%d, %d, %d, %d, %d, ", confId++, num_neurons[n1], num_neurons[n2], epochs[ep], it);

                    printd(LITE, "Creating network.\n");
                    if (n2>0) {
                        ann = fann_create_standard(num_layers, num_input, num_neurons[n1], num_neurons[n2], num_output);
                    } else {
                        ann = fann_create_standard(num_layers-1, num_input, num_neurons[n1], num_output);
                    }

                    train_data = fann_read_train_from_file("../NPUdatasets/rat_spike.train");

                    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
                    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

                    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT); // not sure if we need to use this
                    fann_set_bit_fail_limit(ann, 0.01f);

                    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP); // leave it for now...

                    fann_init_weights(ann, train_data); // is this function deterministic???

                    printd(LITE, "Training network.\n");
                    fann_train_on_data(ann, train_data, epochs[ep], epochs_between_reports, desired_error);

                    printd(LITE, "Testing network on training data. %f\n", fann_test_data(ann, train_data));
                    miss = 0;
                    for(i = 0; i < fann_length_train_data(train_data); i++)
                    {
                        calc_out = fann_run(ann, train_data->input[i]);
                        if(fann_abs(calc_out[0] - train_data->output[i][0])>0.5) {
                            miss ++;
                            printd(HIGH, "%f, should be %f, difference=%f\n",
                                   calc_out[0], train_data->output[i][0],
                                   fann_abs(calc_out[0] - train_data->output[i][0]));
                        }
                    }
                    // Compute miss rate
                    train_miss_rate = (float)miss/fann_length_train_data(train_data)*100;

                    // Validate with test data
                    test_data = fann_read_train_from_file("../NPUdatasets/rat_spike.test");

                    printd(LITE, "Testing network on test data. %f\n", fann_test_data(ann, test_data));
                    miss = 0;
                    for(i = 0; i < fann_length_train_data(test_data); i++)
                    {
                        calc_out = fann_run(ann, test_data->input[i]);
                        if(fann_abs(calc_out[0] - test_data->output[i][0])>0.5) {
                            miss ++;
                            printd(HIGH, "%f, should be %f, difference=%f\n",
                                   calc_out[0], test_data->output[i][0],
                                   fann_abs(calc_out[0] - test_data->output[i][0]));
                        }
                    }
                    test_miss_rate = (float)miss/fann_length_train_data(test_data)*100;


                    // Report on results
                    printd(NONE, "Training data miss rate: %.3f%%\n", train_miss_rate);
                    printd(NONE, "Training data MSE:       %.3f%%\n", fann_test_data(ann, train_data)*100);
                    printd(NONE, "Test data miss rate:     %.3f%%\n", test_miss_rate);
                    printd(NONE, "Test data MSE:           %.3f%%\n", fann_test_data(ann, test_data)*100);
                    // Print to file
                    fprintf(pFile, "%f, %f, %f, %f, %u\n", train_miss_rate, fann_test_data(ann, train_data)*100, test_miss_rate, fann_test_data(ann, test_data)*100, npu_latency[n2][n1]);


                    printd(LITE, "Saving network.\n");
                    sprintf(confIdStr, "%04d", confId);
                    strcpy(dstFilePath, "ann/rat_spike_");
                    strcat(dstFilePath, confIdStr);
                    fann_save(ann, dstFilePath);

                    printd(LITE, "Cleaning up.\n");
                    fann_destroy(ann);
                    fann_destroy_train(train_data);
                    fann_destroy_train(test_data);
                }
            }
        }
    }

    fclose(pFile);

    return 0;
}


