# nntune

1. Type `make` to compile the FANN library and our client tools. Or if you want to do the steps individually:
    * `make fann` builds FANN with CMake
    * `make train` builds the `train` tool (which links against the FANN dynamic library)
    * `make recall` builds the `recall` tool for testing (also links to FANN, of course)
