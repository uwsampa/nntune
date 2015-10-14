#include <stdio.h>
#include "fann.h"

/*
arguments:
 - NN spec filename
 - data filename
*/
int main(int argc, char **argv)
{
    // Argument validation
    if (argc != 4) {
        printf("Error: Incorrect number of arguments!\n");
        printf("  usage:   ./recall <nn file> <dataset> <mode {0:MSE, 1:Classification}>\n");
        printf("  example: ./recall func.nn train.data 0\n");
        return 0;
    }

    struct fann *ann = fann_create_from_file(argv[1]);
    struct fann_train_data *data = fann_read_train_from_file(argv[2]);
    int error_mode = atoi(argv[3]);

    unsigned i, j;
    unsigned errors;
    unsigned count = fann_length_train_data(data);
    unsigned num_outputs = fann_num_output_train_data(data);
    // Squared error
    fann_type se = 0.0;
    // Classification error
    int ce = 0;
    for(i = 0; i < count; ++i)
    {
        fann_type *calc_out = fann_run(ann, data->input[i]);
        for (j = 0; j < num_outputs; ++j) {
            fann_type pred = calc_out[j];
            fann_type actual = data->output[i][j];
            fann_type error = pred - actual;
            se += (error * error);
            if((pred>=0.5 && actual<0.5)||(pred<0.5 && actual>=0.5)) {
                ce ++;
            }
        }
   }
    fann_type mse = se / (count*num_outputs);
    fann_type rmse = sqrt(mse);
    fann_type class_error = (fann_type) ce/count;
    if (error_mode==0) {
        printf("%f\n", fann_test_data(ann, data));
    } else {
        printf("%f\n", class_error);
    }
    fann_destroy(ann);
    return 0;
}
