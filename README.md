# nntune

0. Clone the repository to your Unix system. Use the submodules flag to grab the project's dependencies:
    * `git clone --recurse-submodules git@github.com:sampsyo/cluster-workers.git`
    * Don't forget to switch to this branch: `git checkout -b thierry`

1. If you're building this repo for the first time, install the dependencies by typing in `make install` (requires sudo access)

2. Type `make` to compile the FANN library and our client tools. Or if you want to do the steps individually:
    * `make fann` builds FANN with CMake
    * `make train` builds the `train` tool (which links against the FANN dynamic library)
    * `make recall` builds the `recall` tool for testing (also links to FANN, of course)

3. To run neural network training locally enter the following command:
    * `python nntune.py -train example/jmeint.data` will train multiple neural networks on your local machine and select the best one

4. To run the neural network training on the cluster (need a sampa account)
    * SSH into a sampa machine `ssh sampa-gw.cs.washington.edu`
    * Given the number of workers you wish to parallelize your jobs amongst, enter the following command `python nntune.py -train example/jmeint.data -c <number of workers>`


