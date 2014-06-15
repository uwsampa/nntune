#include <stdio.h>
#include "fann.h"

/*
arguments:
 - NN spec filename
 - data filename
*/
int main(int argc, char **argv)
{
    struct fann *ann = fann_create_from_file(argv[1]);
    struct fann_train_data *data = fann_read_train_from_file(argv[2]);

    unsigned errors;
    unsigned count = fann_length_train_data(data);
    unsigned num_outputs = fann_num_output_train_data(data);
    fann_type se = 0.0;
    for(unsigned i = 0; i < count; ++i)
    {
        fann_type *calc_out = fann_run(ann, data->input[i]);
        for (unsigned j = 0; j < num_outputs; ++j) {
            fann_type value = calc_out[j];
            // printf("%f ", calc_out[j]);
            se += value * value;
        }
        // printf("\n");
    }
    fann_type mse = se / (count * num_outputs);
    fann_type rmse = sqrt(mse);
    printf("%f\n", rmse);

    fann_destroy(ann);
    return 0;
}
