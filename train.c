#include "fann.h"

int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
	unsigned int max_epochs, unsigned int epochs_between_reports, 
	float desired_error, unsigned int epochs)
{
	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
	return 0;
}

int main(int argc, char **argv)
{
	fann_type *calc_out;
	const unsigned int num_input                = 18;
	const unsigned int num_output               = 2;
	const unsigned int num_layers               = 4;
	const unsigned int num_neurons_hidden[4]    = {18,64,2,2};
	const float desired_error                   = (const float) 0.000001;
    const unsigned int max_epochs               = 2000;
	const unsigned int epochs_between_reports   = 10;
	struct fann *ann;
	struct fann_train_data *data;

	unsigned int i = 0;
	unsigned int decimal_point;

	printf("Creating network.\n");
	ann = fann_create_standard_array(num_layers, num_neurons_hidden);

	data = fann_read_train_from_file(argv[1]);

	fann_set_activation_steepness_hidden(ann, 0.5);
	fann_set_activation_steepness_output(ann, 0.5);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID);
	fann_set_activation_function_output(ann, FANN_SIGMOID);
	
	//fann_set_learning_rate(ann, 0.010000);

	//fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	//fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_init_weights(ann, data);
	
	printf("Training network.\n");
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

	printf("Testing network. %f\n", fann_test_data(ann, data));

	for(i = 0; i < fann_length_train_data(data); i++)
	{
		calc_out = fann_run(ann, data->input[i]);
		/*printf("JPEG test (%f,%f) -> %f, should be %f, difference=%f\n",
			   data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
			   fann_abs(calc_out[0] - data->output[i][0]));*/
	}
	
	printf("RMSE = %f\n", sqrt(fann_get_MSE(ann)));

	printf("Saving network.\n");

	fann_save(ann, "jmeint.nn");

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);

	return 0;
}
