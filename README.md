# nntune

1. Type `make` to compile the FANN library and our client tools. Or if you want to do the steps individually:
    * `make fann` builds FANN with CMake (will request sudo permission for installation)
    * Ensure the libraries are added to your path: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`
    * `make train` builds the `train` tool (which links against the FANN dynamic library)
    * `make recall` builds the `recall` tool for testing (also links to FANN, of course)

2. To run neural network training locally enter the following command:
    * `python nntune.py -train test/jmeint.data` will train multiple neural networks on your local machine and select the best one
