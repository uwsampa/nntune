#include "fann.h"
#include <stdlib.h>
#include <string.h>

const unsigned int MAX_LAYERS = 16;
const float DESIRED_ERROR = 0.000001;

int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
	unsigned int max_epochs, unsigned int epochs_between_reports, 
	float desired_error, unsigned int epochs)
{
	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
	return 0;
}

/*
arguments (all required):
 - data filename
 - topology, as number of neurons per layer separated by dashes
 - epochs (integer)
 - learning rate (0.0-1.0 float)
*/
int main(int argc, char **argv)
{
    // Argument 1: data filename.
    const char *datafn = argv[1];

    // Argument 2: topology.
    unsigned int layer_sizes[MAX_LAYERS];
    unsigned int num_layers = 0;
    char *token = strtok(argv[2], "-");
    while (token != NULL) {
        layer_sizes[num_layers] = atoi(token);
        ++num_layers;
        token = strtok(NULL, "-");
    }

    // Argument 3: epoch count.
    unsigned int max_epochs = atoi(argv[3]);

    // Argument 4: learning rate.
    float learning_rate = atof(argv[4]);

    struct fann *ann;
	ann = fann_create_standard_array(num_layers, layer_sizes);

    // Misc parameters.
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
	fann_set_activation_steepness_hidden(ann, 0.5);
	fann_set_activation_steepness_output(ann, 0.5);
	fann_set_activation_function_hidden(ann, FANN_SIGMOID);
	fann_set_activation_function_output(ann, FANN_SIGMOID);
    //fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
    //fann_set_bit_fail_limit(ann, 0.01f);

    struct fann_train_data *data;
    data = fann_read_train_from_file(datafn);
	fann_init_weights(ann, data);
	
    fann_set_learning_rate(ann, learning_rate);
	fann_train_on_data(
        ann,
        data,
        max_epochs,
        10,  // epochs between reports
        DESIRED_ERROR
    );

	printf("Testing network. %f\n", fann_test_data(ann, data));

    fann_type *calc_out;
	for(unsigned int i = 0; i < fann_length_train_data(data); ++i)
	{
		calc_out = fann_run(ann, data->input[i]);
	}
	
	printf("RMSE = %f\n", sqrt(fann_get_MSE(ann)));

	fann_save(ann, "trained.nn");

	fann_destroy_train(data);
	fann_destroy(ann);

	return 0;
}
