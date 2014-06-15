#include <stdio.h>
#include "fann.h"

int main(int argc, char **argv)
{
    struct fann *ann = fann_create_from_file(argv[1]);

    struct fann_train_data *data = fann_read_train_from_file(argv[2]);
    unsigned errors;
    unsigned numSamples = fann_length_train_data(data);
    for(unsigned i = 0; i < numSamples; i++)
    {
        fann_type *calc_out = fann_run(ann, data->input[i]);

        int calc_decision = calc_out[0] > calc_out[1];
        int precise_decision = data->output[i][0] > data->output[i][1];
        if (calc_decision != precise_decision)
            ++errors;
    }
    printf("error rate: %i/%i = %f\n", errors, numSamples, (float)errors / numSamples);

    fann_destroy(ann);
    return 0;
}
